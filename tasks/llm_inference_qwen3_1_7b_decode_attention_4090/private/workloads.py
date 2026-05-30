from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
DTYPE_NAME = "bfloat16"


@dataclass(frozen=True)
class DecodeWorkloadSpec:
    name: str
    seq_lens: tuple[int, ...]
    max_kv_splits: int
    split_tile_size: int | None
    kv_layout: str = "contiguous"
    seed: int = 0
    logit_cap: float = 0.0

    @property
    def batch_size(self) -> int:
        return len(self.seq_lens)

    def to_public_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "batch_size": self.batch_size,
            "seq_lens": list(self.seq_lens),
            "max_kv_splits": self.max_kv_splits,
            "split_tile_size": self.split_tile_size,
            "kv_layout": self.kv_layout,
            "logit_cap": self.logit_cap,
        }


@dataclass
class DecodeInputs:
    q: Any
    k_buffer: Any
    v_buffer: Any
    o: Any
    kv_indptr: Any
    kv_indices: Any
    attn_logits: Any
    attn_lse: Any
    num_kv_splits: Any
    max_kv_splits: int
    sm_scale: float
    k_scale: float
    v_scale: float
    logit_cap: float
    sinks: Any
    xai_temperature_len: int
    has_mla: bool
    use_pdl: bool

    def clone(self) -> "DecodeInputs":
        return DecodeInputs(
            q=self.q.clone(),
            k_buffer=self.k_buffer.clone(),
            v_buffer=self.v_buffer.clone(),
            o=self.o.clone(),
            kv_indptr=self.kv_indptr.clone(),
            kv_indices=self.kv_indices.clone(),
            attn_logits=self.attn_logits.clone(),
            attn_lse=self.attn_lse.clone(),
            num_kv_splits=self.num_kv_splits.clone(),
            max_kv_splits=self.max_kv_splits,
            sm_scale=self.sm_scale,
            k_scale=self.k_scale,
            v_scale=self.v_scale,
            logit_cap=self.logit_cap,
            sinks=None if self.sinks is None else self.sinks.clone(),
            xai_temperature_len=self.xai_temperature_len,
            has_mla=self.has_mla,
            use_pdl=self.use_pdl,
        )

    def readonly_tensors(self) -> dict[str, Any]:
        tensors = {
            "q": self.q,
            "k_buffer": self.k_buffer,
            "v_buffer": self.v_buffer,
            "kv_indptr": self.kv_indptr,
            "kv_indices": self.kv_indices,
            "num_kv_splits": self.num_kv_splits,
        }
        if self.sinks is not None:
            tensors["sinks"] = self.sinks
        return tensors


def build_workloads(profile: str) -> list[DecodeWorkloadSpec]:
    profile = (profile or "standard").strip().lower()
    public = [
        DecodeWorkloadSpec(
            name="decode_b1_s64_split1_contig",
            seq_lens=(64,),
            max_kv_splits=1,
            split_tile_size=None,
            kv_layout="contiguous",
            seed=101,
        ),
        DecodeWorkloadSpec(
            name="decode_b2_s127_251_split2_ragged",
            seq_lens=(127, 251),
            max_kv_splits=2,
            split_tile_size=160,
            kv_layout="permuted",
            seed=102,
        ),
        DecodeWorkloadSpec(
            name="decode_b4_s128_256_384_512_split4_strided",
            seq_lens=(128, 256, 384, 512),
            max_kv_splits=4,
            split_tile_size=192,
            kv_layout="strided",
            seed=103,
        ),
    ]
    if profile in {"public", "verify"}:
        return public
    if profile == "quick":
        return public + [
            DecodeWorkloadSpec(
                name="decode_b8_s256_split4_contig",
                seq_lens=(256,) * 8,
                max_kv_splits=4,
                split_tile_size=192,
                kv_layout="contiguous",
                seed=201,
            ),
        ]
    standard = [
        DecodeWorkloadSpec(
            name="decode_b1_s512_split1_contig",
            seq_lens=(512,),
            max_kv_splits=1,
            split_tile_size=None,
            kv_layout="contiguous",
            seed=301,
        ),
        DecodeWorkloadSpec(
            name="decode_b4_s512_768_1024_1536_split8_ragged",
            seq_lens=(512, 768, 1024, 1536),
            max_kv_splits=8,
            split_tile_size=384,
            kv_layout="permuted",
            seed=302,
        ),
        DecodeWorkloadSpec(
            name="decode_b8_s256_to_1024_split8_strided",
            seq_lens=(256, 384, 512, 640, 768, 896, 1024, 1024),
            max_kv_splits=8,
            split_tile_size=384,
            kv_layout="strided",
            seed=303,
        ),
        DecodeWorkloadSpec(
            name="decode_b2_s2048_4096_split16_contig",
            seq_lens=(2048, 4096),
            max_kv_splits=16,
            split_tile_size=512,
            kv_layout="contiguous",
            seed=304,
        ),
        DecodeWorkloadSpec(
            name="decode_b1_s8192_split16_permuted",
            seq_lens=(8192,),
            max_kv_splits=16,
            split_tile_size=768,
            kv_layout="permuted",
            seed=305,
        ),
    ]
    if profile == "expanded":
        return standard + [
            DecodeWorkloadSpec(
                name="decode_b16_s128_to_2048_split16_strided",
                seq_lens=tuple(128 + 128 * i for i in range(16)),
                max_kv_splits=16,
                split_tile_size=512,
                kv_layout="strided",
                seed=401,
            ),
            DecodeWorkloadSpec(
                name="decode_b1_s32768_split32_contig",
                seq_lens=(32768,),
                max_kv_splits=32,
                split_tile_size=1024,
                kv_layout="contiguous",
                seed=402,
            ),
        ]
    return standard


def repeat_counts(profile: str) -> tuple[int, int]:
    profile = (profile or "standard").strip().lower()
    if profile in {"public", "verify"}:
        return 3, 8
    if profile == "quick":
        return 5, 20
    if profile == "expanded":
        return 5, 30
    return 8, 50


def materialize_workload(spec: DecodeWorkloadSpec, *, torch: Any, device: Any) -> DecodeInputs:
    generator = torch.Generator(device=device)
    generator.manual_seed(spec.seed)
    batch = spec.batch_size
    total_kv = int(sum(spec.seq_lens))
    pool_size = _pool_size(total_kv, spec.kv_layout)

    q = torch.randn(
        (batch, Q_HEADS, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    k_buffer = torch.randn(
        (pool_size, KV_HEADS, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    v_buffer = torch.randn(
        (pool_size, KV_HEADS, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    o = torch.empty((batch, Q_HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16)
    o.fill_(float("nan"))
    attn_logits = torch.empty(
        (batch, Q_HEADS, spec.max_kv_splits, HEAD_DIM),
        device=device,
        dtype=torch.float32,
    )
    attn_lse = torch.empty((batch, Q_HEADS, spec.max_kv_splits), device=device, dtype=torch.float32)
    attn_logits.fill_(float("nan"))
    attn_lse.fill_(float("nan"))

    kv_indices = _make_kv_indices(spec, total_kv=total_kv, pool_size=pool_size, torch=torch, device=device)
    kv_indptr = torch.zeros((batch + 1,), device=device, dtype=torch.int32)
    kv_indptr[1:] = torch.tensor(_cumsum(spec.seq_lens), device=device, dtype=torch.int32)
    num_kv_splits = torch.tensor(
        [_num_splits(seq_len, spec.max_kv_splits, spec.split_tile_size) for seq_len in spec.seq_lens],
        device=device,
        dtype=torch.int32,
    )

    return DecodeInputs(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        o=o,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        attn_logits=attn_logits,
        attn_lse=attn_lse,
        num_kv_splits=num_kv_splits,
        max_kv_splits=spec.max_kv_splits,
        sm_scale=1.0 / math.sqrt(float(HEAD_DIM)),
        k_scale=1.0,
        v_scale=1.0,
        logit_cap=spec.logit_cap,
        sinks=None,
        xai_temperature_len=-1,
        has_mla=False,
        use_pdl=False,
    )


def _pool_size(total_kv: int, layout: str) -> int:
    if layout == "strided":
        return total_kv * 2 + 17
    if layout == "permuted":
        return total_kv + max(97, total_kv // 7)
    return total_kv


def _make_kv_indices(spec: DecodeWorkloadSpec, *, total_kv: int, pool_size: int, torch: Any, device: Any) -> Any:
    if spec.kv_layout == "contiguous":
        return torch.arange(total_kv, device=device, dtype=torch.int64)
    if spec.kv_layout == "strided":
        return (torch.arange(total_kv, device=device, dtype=torch.int64) * 2 + 7) % pool_size

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(spec.seed + 100_000)
    perm = torch.randperm(pool_size, generator=cpu_generator, dtype=torch.int64)
    return perm[:total_kv].to(device=device)


def _num_splits(seq_len: int, max_kv_splits: int, split_tile_size: int | None) -> int:
    if split_tile_size is None:
        return max(1, min(max_kv_splits, 1))
    return max(1, min(max_kv_splits, (seq_len + split_tile_size - 1) // split_tile_size))


def _cumsum(values: tuple[int, ...]) -> list[int]:
    total = 0
    out: list[int] = []
    for value in values:
        total += int(value)
        out.append(total)
    return out
