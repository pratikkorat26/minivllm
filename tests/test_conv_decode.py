"""Compare the compiled one-token path with the general convolution."""

import copy
import os

import pytest
import torch

from lfm3.conv import ShortConv


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_compiled_decode_matches_conv1d_and_preserves_state(device, dtype, bias):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(31)
    reference = ShortConv(16, kernel_size=7, bias=bias, dtype=dtype).to(device).eval()
    # Trace quickly by default; opt into Inductor to validate generated code.
    backend = os.environ.get("LFM3_TEST_COMPILE_BACKEND", "eager")
    compiled = torch.compile(copy.deepcopy(reference), backend=backend, fullgraph=True)
    expected_state = torch.randn(5, 16, 6, device=device, dtype=dtype)
    actual_state = expected_state.clone()
    slots = torch.tensor([3, 0, 4], device=device)
    with torch.inference_mode():
        for _ in range(4):
            x = torch.randn(3, 1, 16, device=device, dtype=dtype)
            expected = reference(x, conv_state=expected_state, slots=slots)
            actual = compiled(x, conv_state=actual_state, slots=slots)
            tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-6
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)


def test_triton_fused_kernels_match_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("triton")
    import torch.nn.functional as F
    from lfm3.kernels import triton_rms_norm, triton_short_conv_decode, triton_swiglu

    torch.manual_seed(0)
    device = torch.device("cuda")
    x = torch.randn(2, 1, 64, device=device, dtype=torch.bfloat16)
    weight = torch.randn(64, device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(
        triton_rms_norm(x, weight, 1e-5), F.rms_norm(x, (64,), weight, 1e-5), atol=2e-2, rtol=1e-2
    )

    gate = torch.randn(2, 1, 64, device=device, dtype=torch.bfloat16)
    up = torch.randn(2, 1, 64, device=device, dtype=torch.bfloat16)
    torch.testing.assert_close(
        triton_swiglu(torch.cat([gate, up], dim=-1)),
        F.silu(gate) * up,
        atol=2e-2,
        rtol=2e-2,
    )

    conv = ShortConv(64, kernel_size=3, bias=True, dtype=torch.bfloat16).to(device).eval()
    ref_state = torch.randn(4, 64, 2, device=device, dtype=torch.bfloat16)
    act_state = ref_state.clone()
    slots = torch.tensor([2, 0], device=device)
    with torch.inference_mode():
        expected = conv(x, conv_state=ref_state, slots=slots)
        y = triton_short_conv_decode(
            conv.in_proj(x), act_state, slots, conv.conv.weight, conv.conv.bias
        )
        actual = conv.out_proj(y)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(act_state, ref_state, atol=0, rtol=0)
