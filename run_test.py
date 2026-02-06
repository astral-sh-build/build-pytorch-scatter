"""
Modal test script for PyTorch Scatter wheel builds.

This script tests a single wheel from GitHub Actions on GPU-enabled machines across
all supported CUDA versions for that PyTorch version. The wheel is downloaded during
the image build using the GITHUB_TOKEN.

Run with:
    WHEEL_NAME=<name> GITHUB_OWNER=<owner> GITHUB_REPO=<repo> GITHUB_RUN_ID=<run_id> GITHUB_TOKEN=<token> uv run --with modal modal run run_test.py

NOTE: Modal currently only supports x86_64 architecture GPUs (NVIDIA A10G, A100, H100, etc.).
If you're testing an aarch64 wheel, you'll need to build an x86_64 version first.
"""

import os
import re
import sys

import modal

# GitHub Actions information (from environment variables)
WHEEL_NAME = os.environ.get("WHEEL_NAME")
GITHUB_OWNER = os.environ.get("GITHUB_OWNER")
GITHUB_REPO = os.environ.get("GITHUB_REPO")
GITHUB_RUN_ID = os.environ.get("GITHUB_RUN_ID")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

if not WHEEL_NAME:
    raise ValueError("WHEEL_NAME environment variable must be set")
if not GITHUB_OWNER:
    raise ValueError("GITHUB_OWNER environment variable must be set")
if not GITHUB_REPO:
    raise ValueError("GITHUB_REPO environment variable must be set")
if not GITHUB_RUN_ID:
    raise ValueError("GITHUB_RUN_ID environment variable must be set")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable must be set")

# CUDA versions to test for each PyTorch major.minor version
def parse_wheel_filename(wheel_path: str) -> dict[str, str]:
    """Parse the wheel filename to extract build information.

    Supports torch_scatter wheel naming pattern:
    torch_scatter-2.1.2+cu.12.4.torch.2.4-cp312-cp312-manylinux...whl
    """
    pattern = r"torch_scatter-(?P<version>[^+]+)\+cu\.(?P<cuda_ver>[\d.]+)\.torch\.(?P<torch_ver>[\d.]+)-cp(?P<py_ver>\d+)-"
    match = re.search(pattern, wheel_path)

    if not match:
        raise ValueError(f"Could not parse wheel filename: {wheel_path}")

    info = match.groupdict()
    torch_ver_parts = info["torch_ver"].split(".")
    info["torch_xy"] = f"{torch_ver_parts[0]}.{torch_ver_parts[1]}"
    return info


def get_pytorch_cuda_index_url(cuda_version: str) -> str:
    """Get the PyTorch index URL for a specific CUDA version."""
    cuda_suffix = cuda_version.replace(".", "")
    return f"https://download.pytorch.org/whl/cu{cuda_suffix}"


# Parse wheel information
wheel_info = parse_wheel_filename(WHEEL_NAME)
torch_version = wheel_info["torch_ver"]
torch_xy = wheel_info["torch_xy"]
build_cuda_version = wheel_info["cuda_ver"]
py_ver_num = wheel_info["py_ver"]
python_version = f"{py_ver_num[0]}.{py_ver_num[1:]}"

# Test against the CUDA version the wheel was built with.
cuda_versions_to_test = [build_cuda_version]

print(f"Setting up tests for {WHEEL_NAME}")
print(f"  PyTorch {torch_version}, testing with CUDA versions: {cuda_versions_to_test}")

# Create a single app
app = modal.App("test-pytorch-scatter")


# Create a function that will download the wheel
def download_wheel_func():
    import io
    import os
    import zipfile

    import requests

    # GitHub API settings
    GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
    GITHUB_OWNER = os.environ["GITHUB_OWNER"]
    GITHUB_REPO = os.environ["GITHUB_REPO"]
    GITHUB_RUN_ID = os.environ["GITHUB_RUN_ID"]
    WHEEL_NAME = os.environ["WHEEL_NAME"]

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    # Get artifacts for this run (paginated)
    artifact = None
    page = 1
    while True:
        artifacts_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/actions/runs/{GITHUB_RUN_ID}/artifacts?per_page=100&page={page}"
        response = requests.get(artifacts_url, headers=headers)
        response.raise_for_status()

        artifacts = response.json()["artifacts"]
        if not artifacts:
            break

        for a in artifacts:
            if a["name"] == WHEEL_NAME:
                artifact = a
                break

        if artifact:
            break
        page += 1

    if not artifact:
        raise ValueError(f"Could not find artifact {WHEEL_NAME}")

    # Download the artifact
    download_url = artifact["archive_download_url"]
    response = requests.get(download_url, headers=headers, stream=True)
    response.raise_for_status()

    # Extract the wheel from the zip
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        for name in zip_ref.namelist():
            if name.endswith(".whl"):
                with open(f"/{name}", "wb") as f:
                    f.write(zip_ref.read(name))
                print(f"Downloaded: {name}")
                break


# Create images for each CUDA version
images = {}
for cuda_version in cuda_versions_to_test:
    images[cuda_version] = (
        modal.Image.debian_slim(python_version=python_version)
        .apt_install("git")
        .pip_install(
            f"torch=={torch_version}",
            index_url=get_pytorch_cuda_index_url(cuda_version),
        )
        .pip_install("packaging", "ninja", "requests")
        .env({
            "GITHUB_TOKEN": GITHUB_TOKEN,
            "GITHUB_OWNER": GITHUB_OWNER,
            "GITHUB_REPO": GITHUB_REPO,
            "GITHUB_RUN_ID": GITHUB_RUN_ID,
            "WHEEL_NAME": WHEEL_NAME,
        })
        .run_function(download_wheel_func)
        .pip_install(f"/{WHEEL_NAME}")
    )


def _run_pytorch_scatter_test(cuda_version: str):
    """Core test logic for PyTorch Scatter - parameterized by CUDA version."""
    import subprocess

    import torch

    print("=" * 80)
    print(f"Testing with CUDA {cuda_version}")
    print("=" * 80)
    print()

    print("=" * 80)
    print("GPU Information")
    print("=" * 80)
    subprocess.run(["nvidia-smi"], check=True)
    print()

    print("=" * 80)
    print("Python and Package Versions")
    print("=" * 80)
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device capability: {torch.cuda.get_device_capability(0)}")
    print()

    print("=" * 80)
    print("Testing PyTorch Scatter")
    print("=" * 80)

    import torch_scatter

    print(f"torch_scatter version: {torch_scatter.__version__}")
    print("✓ torch_scatter imported")

    from torch_scatter import scatter, scatter_max, scatter_mean, segment_csr

    print("✓ Imported scatter functions")

    # Test scatter_mean on CUDA
    src = torch.randn(10, 64, device="cuda")
    index = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 3, 3], device="cuda")
    out = scatter_mean(src, index, dim=0)
    assert out.shape == (4, 64), f"Expected shape (4, 64), got {out.shape}"
    print(f"✓ scatter_mean works: input {src.shape} -> output {out.shape}")

    # Test scatter (sum) on CUDA
    out_sum = scatter(src, index, dim=0, reduce="sum")
    assert out_sum.shape == (4, 64), f"Expected shape (4, 64), got {out_sum.shape}"
    print(f"✓ scatter (sum) works: input {src.shape} -> output {out_sum.shape}")

    # Test scatter_max on CUDA
    out_max, argmax = scatter_max(src, index, dim=0)
    assert out_max.shape == (4, 64), f"Expected shape (4, 64), got {out_max.shape}"
    assert argmax.shape == (4, 64), f"Expected argmax shape (4, 64), got {argmax.shape}"
    print(f"✓ scatter_max works: input {src.shape} -> output {out_max.shape}")

    # Test segment_csr on CUDA
    indptr = torch.tensor([0, 2, 5, 7, 10], device="cuda")
    out_seg = segment_csr(src, indptr, reduce="mean")
    assert out_seg.shape == (4, 64), f"Expected shape (4, 64), got {out_seg.shape}"
    print(f"✓ segment_csr works: input {src.shape} -> output {out_seg.shape}")

    # Verify results are numerically reasonable
    # scatter_mean of identical values should return those values
    src_ones = torch.ones(6, device="cuda")
    idx = torch.tensor([0, 0, 1, 1, 2, 2], device="cuda")
    result = scatter_mean(src_ones, idx, dim=0)
    assert torch.allclose(result, torch.ones(3, device="cuda")), "scatter_mean of ones should be ones"
    print("✓ Numerical correctness verified")

    print()
    print("=" * 80)
    print(f"All tests passed for CUDA {cuda_version}!")
    print("=" * 80)

    return {
        "status": "success",
        "cuda_test_version": cuda_version,
        "pytorch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "gpu_name": str(torch.cuda.get_device_name(0))
        if torch.cuda.is_available()
        else None,
    }


# Static test functions for each CUDA version
# Modal GPU options: "any", "a10g", "a100", "h100", "l4", "l40s"


@app.function(image=images.get("12.1"), gpu="a10g", timeout=600)
def test_cuda121():
    return _run_pytorch_scatter_test("12.1")


@app.function(image=images.get("12.4"), gpu="a10g", timeout=600)
def test_cuda124():
    return _run_pytorch_scatter_test("12.4")


@app.function(image=images.get("12.6"), gpu="a10g", timeout=600)
def test_cuda126():
    return _run_pytorch_scatter_test("12.6")


@app.function(image=images.get("12.8"), gpu="a10g", timeout=600)
def test_cuda128():
    return _run_pytorch_scatter_test("12.8")


@app.function(image=images.get("12.9"), gpu="a10g", timeout=600)
def test_cuda129():
    return _run_pytorch_scatter_test("12.9")


@app.function(image=images.get("13.0"), gpu="a10g", timeout=600)
def test_cuda130():
    return _run_pytorch_scatter_test("13.0")


# Map CUDA versions to their test functions
test_functions = {
    "12.1": test_cuda121,
    "12.4": test_cuda124,
    "12.6": test_cuda126,
    "12.8": test_cuda128,
    "12.9": test_cuda129,
    "13.0": test_cuda130,
}


@app.local_entrypoint()
def main():
    """Main entry point - runs all tests in parallel."""
    print("=" * 80)
    print("PyTorch Scatter Multi-CUDA Version Test Suite")
    print("=" * 80)
    print(f"Wheel: {WHEEL_NAME}")
    print(f"PyTorch version: {torch_version}")
    print(f"Testing against CUDA versions: {cuda_versions_to_test}")
    print("=" * 80)
    print()
    print("Running all tests in parallel...")
    print()

    # Run all tests in parallel
    tasks = []
    for cuda_version in cuda_versions_to_test:
        tasks.append(test_functions[cuda_version].spawn())

    # Wait for all tasks to complete
    results = {}
    for i, cuda_version in enumerate(cuda_versions_to_test):
        try:
            results[cuda_version] = tasks[i].get()
        except Exception as e:
            results[cuda_version] = {"status": "failed", "error": str(e)}

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Wheel: {WHEEL_NAME}")
    print(f"PyTorch: {torch_version}")
    print()

    failed = []
    for cuda_version in cuda_versions_to_test:
        result = results[cuda_version]
        if isinstance(result, Exception):
            status = "✗ FAIL"
            failed.append(cuda_version)
            print(f"CUDA {cuda_version}: {status}")
            print(f"  Error: {result}")
        elif result.get("status") == "success":
            status = "✓ PASS"
            print(f"CUDA {cuda_version}: {status}")
            print(f"  GPU: {result.get('gpu_name', 'unknown')}")
            print(f"  PyTorch CUDA: {result.get('cuda_version', 'unknown')}")
        else:
            status = "✗ FAIL"
            failed.append(cuda_version)
            print(f"CUDA {cuda_version}: {status}")
            print(f"  Error: {result.get('error', 'unknown error')}")

    print("=" * 80)

    if failed:
        print(f"\n❌ {len(failed)} test(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(cuda_versions_to_test)} tests passed!")
