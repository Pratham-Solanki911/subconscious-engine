from __future__ import annotations

import json
import os
import struct
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download, hf_hub_url
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Config,
    Qwen2DecoderLayer,
    Qwen2RotaryEmbedding,
)

from .config import SubconsciousEngineConfig


def _fetch_remote_safetensors_header(url: str, timeout: int = 60) -> Tuple[int, Dict]:
    """Fetch only the 8-byte length prefix and JSON header of a remote safetensors shard."""
    req_len = urllib.request.Request(url, headers={"Range": "bytes=0-7", "User-Agent": "subconscious-engine/0.1"})
    with urllib.request.urlopen(req_len, timeout=timeout) as resp:
        header_len_bytes = resp.read(8)
    header_len = struct.unpack("<Q", header_len_bytes)[0]

    req_hdr = urllib.request.Request(
        url,
        headers={"Range": f"bytes=8-{8 + header_len - 1}", "User-Agent": "subconscious-engine/0.1"},
    )
    with urllib.request.urlopen(req_hdr, timeout=timeout) as resp:
        header_json = json.loads(resp.read(header_len).decode("utf-8"))
    return header_len, header_json


def _download_byte_range_to_file(
    url: str,
    start_byte: int,
    end_byte_inclusive: int,
    out_file,
    chunk_size: int = 16 * 1024 * 1024,
    retries: int = 4,
) -> int:
    """Stream an exact HTTP byte range [start_byte, end_byte_inclusive] directly to an open file."""
    total_expected = end_byte_inclusive - start_byte + 1
    written = 0
    cursor = start_byte

    while cursor <= end_byte_inclusive:
        window_end = min(cursor + chunk_size - 1, end_byte_inclusive)
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "Range": f"bytes={cursor}-{window_end}",
                        "User-Agent": "subconscious-engine/0.1",
                    },
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                out_file.write(data)
                written += len(data)
                cursor += len(data)
                break
            except Exception as exc:
                if attempt + 1 == retries:
                    raise RuntimeError(
                        f"Failed HTTP range fetch {cursor}-{window_end} from {url}: {exc}"
                    ) from exc
                time.sleep(1.5 * (attempt + 1))

    assert written == total_expected, f"Expected {total_expected} bytes, wrote {written}"
    return written


def ensure_surgical_subconscious_shards(
    model_id: str,
    layer_indices: Tuple[int, ...],
    cache_dir: str | Path,
) -> Path:
    """
    Ensure that `cache_dir` contains `config.json`, `model.safetensors.index.json`,
    and safetensors shard(s) containing ONLY the exact tensors for `model.layers.{i}.*`
    for `i in layer_indices`.

    NEVER calls `AutoModelForCausalLM.from_pretrained` and avoids downloading unneeded
    embedding tables, early layers [0..11], late layers [20..27], or LM heads.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    local_index_path = cache_path / "model.safetensors.index.json"
    local_config_path = cache_path / "config.json"

    # 1. Fetch config.json and upstream model.safetensors.index.json
    hub_config_path = Path(hf_hub_download(model_id, "config.json"))
    if not local_config_path.exists():
        local_config_path.write_bytes(hub_config_path.read_bytes())

    hub_index_path = Path(hf_hub_download(model_id, "model.safetensors.index.json"))
    upstream_index = json.loads(hub_index_path.read_text(encoding="utf-8"))
    weight_map: Dict[str, str] = upstream_index["weight_map"]

    prefixes = tuple(f"model.layers.{idx}." for idx in layer_indices)
    required_keys = sorted(k for k in weight_map if k.startswith(prefixes))
    if not required_keys:
        raise ValueError(f"No keys found in {model_id} for layers {layer_indices}")

    # Check if already surgically materialized and complete
    if local_index_path.exists():
        try:
            existing_index = json.loads(local_index_path.read_text(encoding="utf-8"))
            existing_map = existing_index.get("weight_map", {})
            if all(k in existing_map for k in required_keys):
                shard_files = set(existing_map[k] for k in required_keys)
                if all((cache_path / sf).exists() for sf in shard_files):
                    return cache_path
        except Exception:
            pass

    # Group required keys by upstream shard filename
    keys_by_shard: Dict[str, List[str]] = {}
    for k in required_keys:
        shard_name = weight_map[k]
        keys_by_shard.setdefault(shard_name, []).append(k)

    # Check if full upstream shards already exist in local HF cache; if not, surgically slice via HTTP Range
    new_weight_map: Dict[str, str] = {}
    total_bytes = 0

    for shard_name, shard_keys in sorted(keys_by_shard.items()):
        local_shard_path = cache_path / shard_name
        url = hf_hub_url(model_id, shard_name)
        header_len, remote_header = _fetch_remote_safetensors_header(url)
        data_base_offset = 8 + header_len

        # Sort keys by their byte offsets in the remote shard so we can coalesce contiguous byte ranges
        shard_keys_sorted = sorted(
            shard_keys,
            key=lambda k: remote_header[k]["data_offsets"][0],
        )

        # Build new compact header for local_shard_path containing ONLY shard_keys_sorted
        new_header: Dict[str, dict] = {}
        if "__metadata__" in remote_header:
            new_header["__metadata__"] = remote_header["__metadata__"]

        coalesced_spans: List[Tuple[int, int]] = []  # (remote_start, remote_end_exclusive)
        cursor_out = 0
        for k in shard_keys_sorted:
            meta = remote_header[k]
            orig_start, orig_end = meta["data_offsets"]
            tensor_nbytes = orig_end - orig_start
            new_header[k] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "data_offsets": [cursor_out, cursor_out + tensor_nbytes],
            }
            cursor_out += tensor_nbytes
            total_bytes += tensor_nbytes
            new_weight_map[k] = shard_name

            if coalesced_spans and coalesced_spans[-1][1] == orig_start:
                coalesced_spans[-1] = (coalesced_spans[-1][0], orig_end)
            else:
                coalesced_spans.append((orig_start, orig_end))

        header_bytes_raw = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
        # Pad header to 8-byte alignment per safetensors spec
        pad_len = (8 - (len(header_bytes_raw) % 8)) % 8
        header_bytes = header_bytes_raw + (b" " * pad_len)

        tmp_shard_path = local_shard_path.with_suffix(".safetensors.part")
        print(
            f"[MmapSubconscious] Surgically fetching {len(shard_keys_sorted)} tensors "
            f"({cursor_out / (1024**2):.1f} MB) from {shard_name} via {len(coalesced_spans)} contiguous HTTP range(s)..."
        )
        with open(tmp_shard_path, "wb") as f_out:
            f_out.write(struct.pack("<Q", len(header_bytes)))
            f_out.write(header_bytes)
            for span_start, span_end in coalesced_spans:
                abs_start = data_base_offset + span_start
                abs_end_inclusive = data_base_offset + span_end - 1
                _download_byte_range_to_file(url, abs_start, abs_end_inclusive, f_out)

        if local_shard_path.exists():
            local_shard_path.unlink()
        tmp_shard_path.rename(local_shard_path)

    index_payload = {
        "metadata": {"total_size": total_bytes, "surgical_layers": list(layer_indices)},
        "weight_map": new_weight_map,
    }
    local_index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    return cache_path


def _quantize_split_1pct_outlier_99pct_int4(
    weight: torch.Tensor,
    outlier_ratio: float = 0.01,
    group_size: int = 128,
    calib_channel_rms: Optional[torch.Tensor] = None,
    work_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, nn.Module]:
    """
    Split `weight` `[out_features, in_features]` into:
    1. Top 1% Outlier Branch (`w_outlier_fp16` `[out_features, k_out]`, `outlier_idx` `[k_out]`) in 16-bit.
    2. Remaining 99% Bulk Branch quantized to packed symmetric INT4 (`group_size=128`):
       - `w_int4_packed` (`uint8` `[out_features, in_features // 2]`, two signed 4-bit nibbles per byte)
       - `group_scales_fp16` (`float16` `[out_features, in_features // group_size]`)
       - `row_scale_work` (`work_dtype` `[out_features]`): per-row scale normalizing each output channel to unit peak before fast per-tensor AVX-512 VNNI packing
       - `vnni_bulk_linear`: FBGEMM AVX-512 VNNI packed linear kernel initialized with the exact
         group-128 INT4 dequantized bulk weights for hardware-accelerated multi-core CPU matmul.
    """
    out_features, in_features = weight.shape
    k_out = max(1, int(round(outlier_ratio * in_features)))

    w_fp32 = weight.detach().float()
    col_l2 = w_fp32.norm(p=2, dim=0)
    if calib_channel_rms is not None and calib_channel_rms.numel() == in_features:
        saliency = col_l2 * calib_channel_rms.detach().float().to(col_l2.device).clamp(min=1e-5)
    else:
        saliency = col_l2

    _, outlier_idx = torch.topk(saliency, k=k_out, largest=True, sorted=True)
    w_outlier_fp16 = weight[:, outlier_idx].detach().to(dtype=work_dtype).contiguous()

    # Zero out the 1% outlier columns in the 99% bulk matrix so outliers never stretch group scales
    w_bulk = w_fp32.clone()
    w_bulk[:, outlier_idx] = 0.0

    assert in_features % group_size == 0, f"in_features={in_features} must be divisible by group_size={group_size}"
    num_groups = in_features // group_size
    w_grouped = w_bulk.view(out_features, num_groups, group_size)
    group_scales = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / 7.0
    q_int4 = torch.clamp(torch.round(w_grouped / group_scales), -7.0, 7.0)
    w_deq = (q_int4 * group_scales).reshape(out_features, in_features)

    # Pack two signed 4-bit values (biased by +8 into [1..15]) per uint8 byte
    q_u8 = (q_int4.to(torch.int16) + 8).to(torch.uint8).view(out_features, in_features)
    w_int4_packed = ((q_u8[:, 0::2] << 4) | (q_u8[:, 1::2] & 0x0F)).contiguous()
    group_scales_fp16 = group_scales.squeeze(-1).to(dtype=torch.float16).contiguous()

    # Cap only the top 2% outlier rows (row_scale >= 1.0) so they don't stretch the global per-tensor
    # AVX-512 VNNI scale while never amplifying quantization noise on normal/small rows
    row_amax = w_deq.abs().amax(dim=-1).clamp(min=1e-6)
    row_cap = torch.quantile(row_amax, 0.98).clamp(min=1e-6)
    row_scale = (row_amax / row_cap).clamp(min=1.0)
    row_scale_work = row_scale.to(dtype=work_dtype).contiguous()

    tmp_linear = nn.Linear(in_features, out_features, bias=False)
    tmp_linear.weight.data.copy_(w_deq / row_scale.unsqueeze(-1))
    vnni_bulk_linear = torch.ao.quantization.quantize_dynamic(
        nn.Sequential(tmp_linear),
        {nn.Linear},
        dtype=torch.qint8,
    )[0]

    return outlier_idx, w_outlier_fp16, w_int4_packed, group_scales_fp16, row_scale_work, vnni_bulk_linear


class IntraLayerSplitQwen2MLP(nn.Module):
    """
    Intra-Layer Mixed-Precision Split MLP for `Qwen2DecoderLayer`:
    - Keeps top 1% most salient input channels (`~36` of `3584` for `gate_proj`/`up_proj` and
      `~190` of `18944` for `down_proj`) in contiguous 16-bit (`FP16`/`BF16`).
    - Quantizes the remaining 99% bulk columns into symmetric `INT4` (`group_size=128`) with
      fused `gate_up` AVX-512 VNNI multi-threaded CPU execution:
      `y = matmul_int4(x_bulk, W_bulk_int4) + matmul_fp16(x_outlier, W_outlier_fp16)`
    """

    def __init__(
        self,
        orig_mlp: nn.Module,
        outlier_ratio: float = 0.01,
        group_size: int = 128,
        calib_input: Optional[torch.Tensor] = None,
        work_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.orig_mlp = orig_mlp
        self.act_fn = orig_mlp.act_fn
        self.work_dtype = work_dtype
        self.outlier_ratio = outlier_ratio
        self.group_size = group_size
        self.use_mixed_precision: bool = True

        with torch.no_grad():
            gate_w = orig_mlp.gate_proj.weight.detach()
            up_w = orig_mlp.up_proj.weight.detach()
            down_w = orig_mlp.down_proj.weight.detach()

            calib_gu_rms: Optional[torch.Tensor] = None
            calib_down_rms: Optional[torch.Tensor] = None
            if calib_input is not None:
                x_cal = calib_input.detach().float().reshape(-1, gate_w.shape[1])
                calib_gu_rms = x_cal.pow(2).mean(dim=0).sqrt()
                mid_cal = self.act_fn(F.linear(x_cal, gate_w.float())) * F.linear(x_cal, up_w.float())
                calib_down_rms = mid_cal.pow(2).mean(dim=0).sqrt()

            # Fused [gate_proj; up_proj] (shape [37888, 3584]) -> 1% FP16 outliers (36 cols) + 99% INT4 bulk
            gate_up_w = torch.cat([gate_w, up_w], dim=0)
            (
                gu_idx,
                gu_out_fp16,
                gu_i4_pack,
                gu_scales,
                gu_row_scale,
                self.gate_up_bulk_vnni,
            ) = _quantize_split_1pct_outlier_99pct_int4(
                gate_up_w,
                outlier_ratio=outlier_ratio,
                group_size=group_size,
                calib_channel_rms=calib_gu_rms,
                work_dtype=work_dtype,
            )
            self.register_buffer("gate_up_outlier_idx", gu_idx, persistent=False)
            self.register_buffer("gate_up_outlier_fp16", gu_out_fp16, persistent=False)
            self.register_buffer("gate_up_int4_packed", gu_i4_pack, persistent=False)
            self.register_buffer("gate_up_scales_fp16", gu_scales, persistent=False)
            self.register_buffer("gate_up_row_scale", gu_row_scale, persistent=False)

            # down_proj (shape [3584, 18944]) -> 1% FP16 outliers (190 cols) + 99% INT4 bulk
            (
                dn_idx,
                dn_out_fp16,
                dn_i4_pack,
                dn_scales,
                dn_row_scale,
                self.down_bulk_vnni,
            ) = _quantize_split_1pct_outlier_99pct_int4(
                down_w,
                outlier_ratio=outlier_ratio,
                group_size=group_size,
                calib_channel_rms=calib_down_rms,
                work_dtype=work_dtype,
            )
            self.register_buffer("down_outlier_idx", dn_idx, persistent=False)
            self.register_buffer("down_outlier_fp16", dn_out_fp16, persistent=False)
            self.register_buffer("down_int4_packed", dn_i4_pack, persistent=False)
            self.register_buffer("down_scales_fp16", dn_scales, persistent=False)
            self.register_buffer("down_row_scale", dn_row_scale, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_mixed_precision:
            return self.orig_mlp(x)

        orig_dtype = x.dtype
        x_work = x.to(dtype=self.work_dtype)

        # 1. Fused Gate + Up: parallel sum of 99% INT4 bulk + 1% FP16/BF16 outlier columns
        x_gu_outlier = x_work.index_select(-1, self.gate_up_outlier_idx)
        gu_outlier = F.linear(x_gu_outlier, self.gate_up_outlier_fp16)
        x_gu_bulk = x_work.float().clone()
        x_gu_bulk.index_fill_(-1, self.gate_up_outlier_idx, 0.0)
        gu_bulk = self.gate_up_bulk_vnni(x_gu_bulk).to(dtype=self.work_dtype) * self.gate_up_row_scale
        gate_out, up_out = (gu_bulk + gu_outlier).chunk(2, dim=-1)

        hidden = self.act_fn(gate_out) * up_out

        # 2. Down-Proj: parallel sum of 99% INT4 bulk + 1% FP16/BF16 outlier columns
        h_dn_outlier = hidden.index_select(-1, self.down_outlier_idx)
        dn_outlier = F.linear(h_dn_outlier, self.down_outlier_fp16)
        h_dn_bulk = hidden.float().clone()
        h_dn_bulk.index_fill_(-1, self.down_outlier_idx, 0.0)
        dn_bulk = self.down_bulk_vnni(h_dn_bulk).to(dtype=self.work_dtype) * self.down_row_scale
        return (dn_bulk + dn_outlier).to(dtype=orig_dtype)


class MmapSubconsciousCore(nn.Module):
    """
    Zero-Full-Load Subconscious Middle-Layer Engine on System RAM (`cpu`).

    Instantiates ONLY empty `Qwen2DecoderLayer` objects for `config.subconscious_layer_indices`
    (e.g. `[12..19]`) on `torch.device("meta")` and binds ONLY their specific `state_dict` keys
    directly from `safetensors.safe_open(..., framework="pt", device="cpu")` using
    `model.safetensors.index.json`.

    NEVER calls `AutoModelForCausalLM.from_pretrained()`.
    Bypasses the embedding table, early layers [0..11], late layers [20..27], and LM head.
    """

    def __init__(self, config: SubconsciousEngineConfig, auto_fetch: bool = True) -> None:
        super().__init__()
        self.engine_config = config
        self.layer_indices: Tuple[int, ...] = config.subconscious_layer_indices

        if auto_fetch:
            self.shard_dir = ensure_surgical_subconscious_shards(
                model_id=config.subconscious_model_id,
                layer_indices=self.layer_indices,
                cache_dir=config.subconscious_cache_dir,
            )
        else:
            self.shard_dir = Path(config.subconscious_cache_dir)

        config_path = self.shard_dir / "config.json"
        if config_path.exists():
            self.qwen_config: Qwen2Config = Qwen2Config.from_json_file(str(config_path))
        else:
            self.qwen_config = AutoConfig.from_pretrained(config.subconscious_model_id)

        # Ensure eager attention on CPU for deterministic, zero-overhead execution on short k=4 sequences
        self.qwen_config._attn_implementation = "eager"

        # 1. Instantiate ONLY the target Qwen2DecoderLayer blocks on "meta" (zero RAM allocated)
        self.layers = nn.ModuleDict()
        with torch.device("meta"):
            for layer_idx in self.layer_indices:
                layer = Qwen2DecoderLayer(self.qwen_config, layer_idx=layer_idx)
                self.layers[str(layer_idx)] = layer

        # 2. Instantiate RotaryEmbedding on CPU (compatible across transformers 4.x and 5.x)
        try:
            self.rotary_emb = Qwen2RotaryEmbedding(config=self.qwen_config, device=torch.device("cpu"))
        except TypeError:
            self.rotary_emb = Qwen2RotaryEmbedding(config=self.qwen_config).to("cpu")

        # 3. Keep safetensors mmap handles alive and bind each layer's parameters zero-copy on CPU
        torch.set_num_threads(max(1, os.cpu_count() or 4))
        self._mmap_handles: Dict[str, object] = {}
        self.loaded_tensor_count: int = 0
        self.loaded_bytes: int = 0
        self.intra_layer_split_enabled: bool = False
        self._bind_mmap_weights()

        # Freeze all Subconscious CPU parameters
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def enable_intra_layer_split(
        self,
        calib_essence_inputs: Optional[torch.Tensor] = None,
        outlier_ratio: Optional[float] = None,
        group_size: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Convert each middle layer's MLP (`gate_proj`, `up_proj`, `down_proj`) into
        `IntraLayerSplitQwen2MLP` (1% FP16/BF16 Outlier Branch + 99% Packed INT4 Group-128 Bulk Branch),
        while keeping `self_attn` (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and `RMSNorm`s in full 16-bit precision.
        """
        ratio = outlier_ratio if outlier_ratio is not None else self.engine_config.intra_layer_outlier_ratio
        g_size = group_size if group_size is not None else self.engine_config.intra_layer_int4_group_size

        h_cal: Optional[torch.Tensor] = None
        position_ids = None
        position_embeddings = None
        causal_mask = None
        if calib_essence_inputs is not None:
            h_cal = calib_essence_inputs.detach().to(device="cpu", dtype=self.engine_config.dtype)
            bsz, seq_len, _ = h_cal.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device="cpu").unsqueeze(0).expand(bsz, -1)
            position_embeddings = self.rotary_emb(h_cal, position_ids)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), dtype=h_cal.dtype, device="cpu"),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)

        int4_packed_bytes = 0
        outlier_fp16_bytes = 0
        with torch.no_grad():
            for layer_idx in self.layer_indices:
                layer = self.layers[str(layer_idx)]
                if isinstance(layer.mlp, IntraLayerSplitQwen2MLP):
                    layer.mlp.use_mixed_precision = True
                    continue

                mlp_cal_in: Optional[torch.Tensor] = None
                if h_cal is not None:
                    residual = h_cal
                    h_norm1 = layer.input_layernorm(h_cal)
                    attn_out = layer.self_attn(
                        hidden_states=h_norm1,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        use_cache=False,
                    )[0]
                    h_after_attn = residual + attn_out
                    mlp_cal_in = layer.post_attention_layernorm(h_after_attn)
                    h_cal = h_after_attn + layer.mlp(mlp_cal_in)

                split_mlp = IntraLayerSplitQwen2MLP(
                    orig_mlp=layer.mlp,
                    outlier_ratio=ratio,
                    group_size=g_size,
                    calib_input=mlp_cal_in,
                    work_dtype=self.engine_config.dtype,
                )
                layer.mlp = split_mlp
                int4_packed_bytes += (
                    split_mlp.gate_up_int4_packed.numel() * split_mlp.gate_up_int4_packed.element_size()
                    + split_mlp.down_int4_packed.numel() * split_mlp.down_int4_packed.element_size()
                )
                outlier_fp16_bytes += (
                    split_mlp.gate_up_outlier_fp16.numel() * split_mlp.gate_up_outlier_fp16.element_size()
                    + split_mlp.down_outlier_fp16.numel() * split_mlp.down_outlier_fp16.element_size()
                )

        self.intra_layer_split_enabled = True
        return {
            "intra_layer_split_enabled": True,
            "outlier_ratio": ratio,
            "group_size": g_size,
            "gate_up_outlier_channels": int(self.layers[str(self.layer_indices[0])].mlp.gate_up_outlier_idx.numel()),
            "down_outlier_channels": int(self.layers[str(self.layer_indices[0])].mlp.down_outlier_idx.numel()),
            "int4_packed_mlp_mb": round(int4_packed_bytes / (1024 ** 2), 2),
            "outlier_fp16_mlp_mb": round(outlier_fp16_bytes / (1024 ** 2), 2),
        }

    def set_mixed_precision_mode(self, enabled: bool) -> None:
        """Toggle between Intra-Layer Split Mixed-Precision (`True`) and Full 16-bit Baseline (`False`)."""
        self.intra_layer_split_enabled = enabled
        for layer_idx in self.layer_indices:
            layer = self.layers[str(layer_idx)]
            if isinstance(layer.mlp, IntraLayerSplitQwen2MLP):
                layer.mlp.use_mixed_precision = enabled

    def _bind_mmap_weights(self) -> None:
        """Bind ONLY `model.layers.{12..19}.*` parameters via `safetensors.safe_open` on CPU."""
        index_path = self.shard_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing safetensors index: {index_path}")

        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map: Dict[str, str] = index_data["weight_map"]

        # Open only the shards referenced by our target middle layers
        prefixes = tuple(f"model.layers.{idx}." for idx in self.layer_indices)
        needed_shards = sorted({shard for k, shard in weight_map.items() if k.startswith(prefixes)})

        for shard_name in needed_shards:
            shard_file = self.shard_dir / shard_name
            # mytail: Keep safe_open handle alive on self._mmap_handles so OS mmap pages stay valid.
            self._mmap_handles[shard_name] = safe_open(str(shard_file), framework="pt", device="cpu")

        for layer_idx in self.layer_indices:
            prefix = f"model.layers.{layer_idx}."
            layer_module = self.layers[str(layer_idx)]
            layer_state: Dict[str, torch.Tensor] = {}

            for full_key, shard_name in weight_map.items():
                if full_key.startswith(prefix):
                    sub_key = full_key[len(prefix) :]
                    tensor = self._mmap_handles[shard_name].get_tensor(full_key)
                    if tensor.dtype != self.engine_config.dtype:
                        tensor = tensor.to(dtype=self.engine_config.dtype)
                    layer_state[sub_key] = tensor
                    self.loaded_tensor_count += 1
                    self.loaded_bytes += tensor.numel() * tensor.element_size()

            missing, unexpected = layer_module.load_state_dict(layer_state, strict=True, assign=True)
            if missing or unexpected:
                raise RuntimeError(
                    f"Layer {layer_idx} state_dict mismatch: missing={missing}, unexpected={unexpected}"
                )

    def verify_zero_cuda_and_isolation(self) -> Dict[str, object]:
        """Verify that zero parameters reside on CUDA and only the target middle layers exist."""
        cuda_params = [name for name, p in self.named_parameters() if p.device.type != "cpu"]
        meta_params = [name for name, p in self.named_parameters() if p.device.type == "meta"]
        assert not cuda_params, f"Subconscious parameters leaked to CUDA: {cuda_params}"
        assert not meta_params, f"Unbound meta parameters remain: {meta_params}"
        return {
            "layer_indices": self.layer_indices,
            "num_layers": len(self.layers),
            "loaded_tensor_count": self.loaded_tensor_count,
            "loaded_gb": round(self.loaded_bytes / (1024**3), 3),
            "all_cpu": True,
        }

    def forward(
        self,
        essence_hidden_states: torch.Tensor,
        layer_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Execute the memory-mapped CPU middle layers `[12..19]` on the projected essence vectors.

        Args:
            essence_hidden_states: Tensor of shape `[B, k, d_big]` on CPU or GPU.
            layer_mask: Optional binary mask of shape `[L]` or `[B, L]` in `{0, 1}` indicating
                        which Subconscious layers to execute.

        Returns:
            Tuple of `(refined_essence_cpu, executed_layer_count)` where `refined_essence_cpu`
            has shape `[B, k, d_big]` on CPU.
        """
        orig_dtype = essence_hidden_states.dtype
        h_cpu = essence_hidden_states.to(device="cpu", dtype=self.engine_config.dtype)
        bsz, seq_len, _ = h_cpu.shape

        position_ids = torch.arange(seq_len, dtype=torch.long, device="cpu").unsqueeze(0).expand(bsz, -1)
        position_embeddings = self.rotary_emb(h_cpu, position_ids)

        # Causal mask for the k=4 essence sequence in eager attention `[B, 1, k, k]`
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), dtype=h_cpu.dtype, device="cpu"),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0).expand(bsz, 1, seq_len, seq_len)

        if layer_mask is not None:
            mask_cpu = layer_mask.detach().to(device="cpu")
            if mask_cpu.dim() == 2:
                # Aggregate batch mask conservatively (run layer if active for any sample in batch)
                mask_1d = (mask_cpu.float().mean(dim=0) >= 0.5).tolist()
            else:
                mask_1d = (mask_cpu.float() >= 0.5).tolist()
        else:
            mask_1d = [True] * len(self.layer_indices)

        executed_layers = 0
        with torch.no_grad():
            for idx_pos, layer_idx in enumerate(self.layer_indices):
                if not mask_1d[idx_pos]:
                    continue
                layer = self.layers[str(layer_idx)]
                out = layer(
                    hidden_states=h_cpu,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                    use_cache=False,
                )
                h_cpu = out[0] if isinstance(out, tuple) else out
                executed_layers += 1

        return h_cpu.to(dtype=orig_dtype), executed_layers
