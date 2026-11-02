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
