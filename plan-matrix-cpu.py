# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "packaging",
# ]
# ///

import json
import os

from packaging.version import Version

# Add or remove versions as needed based on pytorch-scatter compatibility.
PYTORCH_SCATTER_SUPPORTED_TORCH_VERSIONS = [
    "2.3.1",
    "2.4.1",
    "2.5.1",
    "2.6.0",
    "2.7.1",
    "2.8.0",
    "2.9.0",
    "2.10.0",
]

ARCH_TORCH_PAIRS = {
    "x86_64": ["2.3.1", "2.4.1", "2.5.1", "2.6.0", "2.7.1", "2.8.0", "2.9.0", "2.10.0"],
    "aarch64": ["2.6.0", "2.7.1", "2.8.0", "2.9.0", "2.10.0"],
}

# Supported Python versions for each PyTorch version.
# See: https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
TORCH_PYTHON_SUPPORT = {
    "2.3": ["3.8", "3.9", "3.10", "3.11", "3.12"],
    "2.4": ["3.8", "3.9", "3.10", "3.11", "3.12"],
    "2.5": ["3.9", "3.10", "3.11", "3.12"],
    "2.6": ["3.9", "3.10", "3.11", "3.12"],
    "2.7": ["3.9", "3.10", "3.11", "3.12", "3.13"],
    "2.8": ["3.9", "3.10", "3.11", "3.12", "3.13"],
    "2.9": ["3.10", "3.11", "3.12", "3.13", "3.14"],
    "2.10": ["3.10", "3.11", "3.12", "3.13", "3.14"],
}

# The glibc version to use for each PyTorch version, for manylinux builds.
# See: https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
TORCH_GLIBC_VERSION: dict[str, str] = {
    "2.3": "2_17",
    "2.4": "2_17",
    "2.5": "2_17",
    "2.6": "2_28",
    "2.7": "2_28",
    "2.8": "2_28",
    "2.9": "2_28",
    "2.10": "2_28",
}

AUDITWHEEL_EXCLUDES = [
    "libc10.so",
    "libtorch.so",
    "libtorch_python.so",
    "libtorch_cpu.so",
]

# Matrix exclusions.
EXCLUSIONS = [
    # No exclusions yet.
]


def main() -> None:
    # Every matrix member is a primary 4-tuple of:
    # `torch-version`: the PyTorch version as "X.Y.Z", e.g. "2.7.0"
    # `python-version`: the Python version as "3.X", e.g. "3.10"
    # `cxx11-abi`: "TRUE" or "FALSE"
    # `target-arch`: the target architecture, e.g. "x86_64" or "aarch64"

    rows = []
    for target_arch, torch_versions in ARCH_TORCH_PAIRS.items():
        for torch_version in torch_versions:
            if torch_version not in PYTORCH_SCATTER_SUPPORTED_TORCH_VERSIONS:
                continue

            torch_version_parsed = Version(torch_version)
            torch_x_y = f"{torch_version_parsed.major}.{torch_version_parsed.minor}"
            for python_version in TORCH_PYTHON_SUPPORT[torch_x_y]:
                cxx11_abi = torch_version_parsed >= Version("2.7.0")
                row = {
                    "target-arch": target_arch,
                    "torch-version": str(torch_version_parsed),
                    "python-version": python_version,
                    "cxx11-abi": "TRUE" if cxx11_abi else "FALSE",
                }

                if row not in EXCLUSIONS:
                    rows.append(row)

    # Transform each row to add various nice-to-have representations of fields.
    for row in rows:
        torch_version = Version(row["torch-version"])

        # `CI_*` variables: same as the original ones.
        row["CI_TORCH_VERSION"] = row["torch-version"]
        row["CI_PYTHON_VERSION"] = row["python-version"]

        # `MATRIX_TORCH_VERSION`: `torch-version`, but only X.Y, no patch
        row["MATRIX_TORCH_VERSION"] = f"{torch_version.major}.{torch_version.minor}"

        # `MATRIX_PYTHON_VERSION`: same as `python-version`, but with the dot removed
        row["MATRIX_PYTHON_VERSION"] = row["python-version"].replace(".", "")

        # MANYLINUX_GLIBC_VERSION: the glibc version to use for manylinux builds.
        row["MANYLINUX_GLIBC_VERSION"] = TORCH_GLIBC_VERSION[
            row["MATRIX_TORCH_VERSION"]
        ]

        # `CI_AUDITWHEEL_EXCLUDES`: `--exclude {lib}` for each lib that should
        # be excluded when running `auditwheel repair`.
        row["CI_AUDITWHEEL_EXCLUDES"] = " ".join(
            f"--exclude {lib}" for lib in AUDITWHEEL_EXCLUDES
        )

        # RUNNER: the GitHub Actions runner to use.
        if row["target-arch"] == "x86_64":
            row["RUNNER"] = "depot-ubuntu-24.04-8"
        elif row["target-arch"] == "aarch64":
            row["RUNNER"] = "depot-ubuntu-24.04-arm-8"
        else:
            raise ValueError(f"Unknown target arch: {row['target-arch']}")

    # For PR builds, limit matrix to a single entry for faster CI.
    if os.environ.get("LIMIT_MATRIX") == "1":
        rows = rows[:1]
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
