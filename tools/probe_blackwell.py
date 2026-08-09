#!/usr/bin/env python
"""Decide, on the actual GPU, whether the prebuilt Blackwell stack works.

Run this ON A B300 NODE with a GPU allocated, after installing the stack from
INSTALL_B300.md. It answers the one question we cannot answer off-hardware: whether the
`sm_100` cubins in the prebuilt torch and flash-attn wheels load on this device.

Why this cannot be decided in advance:

  * torch 2.8.0+cu128 `libtorch_cuda.so` carries SASS for sm_70/75/80/86/89/90/90a/100/100a/
    120/120a and NO PTX.
  * the prebuilt flash-attn 2.8.3 wheel carries SASS for sm_80/90/100/120 and NO PTX.
  * neither carries sm_103, and CUDA 12.9 does expose compute_103/103a/103f as distinct
    targets, so "10.3" is a real, separate architecture.

Without PTX there is no JIT fallback: either the driver accepts an sm_100 cubin on this
device under CUDA's minor-version binary compatibility rule, or every kernel launch fails
with "no kernel image is available for execution on the device". Step 4 below settles it by
actually running the attention kernel nanotron's training path uses.

    python tools/probe_blackwell.py

Exit 0 = the prebuilt stack works, install as-is. Exit 3 = rebuild from source (see
INSTALL_B300.md section 4). Exit 2 = something else is wrong; read the message.
"""
from __future__ import annotations

import sys
import traceback

FAIL_HINT = "no kernel image"


def main():
    print("=" * 72)
    print("STEP 1 — device identity")
    print("=" * 72)
    try:
        import torch
    except Exception as e:
        print(f"torch import failed: {e!r}")
        return 2
    if not torch.cuda.is_available():
        print("no CUDA device visible — allocate a GPU first (this must run ON a B300 node)")
        return 2
    cap = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"  device                 : {name}")
    print(f"  compute capability     : {cap[0]}.{cap[1]}   -> sm_{cap[0]}{cap[1]}")
    print(f"  torch                  : {torch.__version__} (cuda {torch.version.cuda})")
    print(f"  torch arch_list        : {torch.cuda.get_arch_list()}")
    print(f"  driver-visible devices : {torch.cuda.device_count()}")
    print(f"  total memory           : {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

    sm = f"sm_{cap[0]}{cap[1]}"
    archs = torch.cuda.get_arch_list()
    if sm in archs:
        print(f"  -> {sm} is compiled into torch directly.")
    else:
        same_major = [a for a in archs if a.startswith(f"sm_{cap[0]}")]
        print(f"  -> {sm} is NOT in torch's arch list. Same-major cubins present: {same_major}")
        print("     This must work by CUDA minor-version binary compatibility, or not at all.")
        print("     Steps 2-4 decide it empirically.")

    print()
    print("=" * 72)
    print("STEP 2 — a real torch CUDA kernel (matmul in bf16)")
    print("=" * 72)
    try:
        a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        c = (a @ a).float().abs().mean().item()
        torch.cuda.synchronize()
        print(f"  OK  bf16 4096^3 matmul ran, mean|C| = {c:.4f}")
    except Exception as e:
        print(f"  FAIL {type(e).__name__}: {e}")
        if FAIL_HINT in str(e):
            print("  -> torch itself has no usable cubin for this device. Rebuild torch, or use a")
            print("     torch build whose TORCH_CUDA_ARCH_LIST includes this arch.")
            return 3
        traceback.print_exc()
        return 2

    print()
    print("=" * 72)
    print("STEP 3 — flash-attn import + build provenance")
    print("=" * 72)
    try:
        import flash_attn
        print(f"  flash_attn {flash_attn.__version__}")
    except Exception as e:
        print(f"  FAIL importing flash_attn: {e!r}")
        return 2

    print()
    print("=" * 72)
    print("STEP 4 — the kernel nanotron actually trains with (flash_attn_varlen_func)")
    print("=" * 72)
    try:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func
        nheads, hdim, slen = 16, 128, 256          # same head dim as the 1.5B config (2048/16)
        q = torch.randn(slen, nheads, hdim, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(slen, nheads, hdim, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(slen, nheads, hdim, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, slen], dtype=torch.int32, device="cuda")
        out = flash_attn_varlen_func(
            q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=slen, max_seqlen_k=slen,
            dropout_p=0.0, softmax_scale=None, causal=True,
        )
        torch.cuda.synchronize()
        print(f"  OK  forward ran, out {tuple(out.shape)} {out.dtype}, mean|out| = {out.float().abs().mean():.4f}")
        # backward too: training needs it and it is a separate set of kernels
        q.requires_grad_(True); k.requires_grad_(True); v.requires_grad_(True)
        out = flash_attn_varlen_func(q=q, k=k, v=v, cu_seqlens_q=cu, cu_seqlens_k=cu,
                                     max_seqlen_q=slen, max_seqlen_k=slen,
                                     dropout_p=0.0, softmax_scale=None, causal=True)
        out.float().sum().backward()
        torch.cuda.synchronize()
        print(f"  OK  backward ran, grad_q mean|.| = {q.grad.float().abs().mean():.4f}")
    except Exception as e:
        print(f"  FAIL {type(e).__name__}: {e}")
        if FAIL_HINT in str(e):
            print()
            print("  -> DIAGNOSIS: flash-attn has no cubin this device can load.")
            print("     The prebuilt wheel is sm_80/90/100/120 with no PTX, so if this device is")
            print("     sm_103 and the driver will not accept an sm_100 cubin, the only fix is to")
            print("     rebuild flash-attn from source. See INSTALL_B300.md section 4.")
            return 3
        traceback.print_exc()
        return 2

    print()
    print("=" * 72)
    print("RESULT: the prebuilt stack works on this device. Install as-is.")
    print(f"Report back: compute capability {cap[0]}.{cap[1]}, {torch.cuda.device_count()} GPUs "
          f"visible, {torch.cuda.get_device_properties(0).total_memory/2**30:.0f} GiB each.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
