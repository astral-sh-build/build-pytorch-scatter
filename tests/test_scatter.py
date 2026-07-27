from importlib.metadata import version

import pytest
import torch
from torch_scatter import scatter, scatter_max, scatter_mean, segment_csr


@pytest.fixture(scope="module")
def device() -> torch.device:
    assert torch.cuda.is_available(), "The tests must run on a CUDA GPU"
    return torch.device("cuda")


def test_published_cuda_wheel(device: torch.device) -> None:
    assert version("torch-scatter") == "2.1.2+cu.12.8.torch.2.10"
    assert torch.__version__ == "2.10.0+cu128"
    assert torch.version.cuda == "12.8"
    assert torch.cuda.get_device_name(device)


def test_scatter_sum(device: torch.device) -> None:
    source = torch.tensor([1.0, 3.0, 2.0, 4.0], device=device)
    index = torch.tensor([0, 0, 1, 1], device=device)

    actual = scatter(source, index, dim=0, reduce="sum")

    torch.testing.assert_close(actual, torch.tensor([4.0, 6.0], device=device))


def test_scatter_mean(device: torch.device) -> None:
    source = torch.tensor([1.0, 3.0, 2.0, 4.0], device=device)
    index = torch.tensor([0, 0, 1, 1], device=device)

    actual = scatter_mean(source, index, dim=0)

    torch.testing.assert_close(actual, torch.tensor([2.0, 3.0], device=device))


def test_scatter_max(device: torch.device) -> None:
    source = torch.tensor([1.0, 3.0, 4.0, 2.0], device=device)
    index = torch.tensor([0, 0, 1, 1], device=device)

    actual, argmax = scatter_max(source, index, dim=0)

    torch.testing.assert_close(actual, torch.tensor([3.0, 4.0], device=device))
    torch.testing.assert_close(argmax, torch.tensor([1, 2], device=device))


def test_segment_csr(device: torch.device) -> None:
    source = torch.tensor([1.0, 3.0, 2.0, 4.0], device=device)
    indptr = torch.tensor([0, 2, 4], device=device)

    actual = segment_csr(source, indptr, reduce="mean")

    torch.testing.assert_close(actual, torch.tensor([2.0, 3.0], device=device))


def test_scatter_backward(device: torch.device) -> None:
    source = torch.tensor([1.0, 3.0, 2.0, 4.0], device=device, requires_grad=True)
    index = torch.tensor([0, 0, 1, 1], device=device)

    scatter(source, index, dim=0, reduce="sum").square().sum().backward()

    assert source.grad is not None
    torch.testing.assert_close(
        source.grad,
        torch.tensor([8.0, 8.0, 12.0, 12.0], device=device),
    )
