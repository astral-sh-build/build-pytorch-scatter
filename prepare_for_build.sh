#!/bin/bash
# Script to prepare the build environment for PyTorch Scatter.
#
# Example usage:
#   ./prepare_for_build.sh v2.1.2

set -euxo pipefail

export ROOT=`pwd`

if [ $# -ne 1 ]; then
    echo "Usage: $0 <pytorch_scatter_version>"
    echo "Example: $0 v2.1.2"
    exit 1
fi

PYTORCH_SCATTER_VERSION=$1

# Ensure that the PyTorch Scatter version is supported.
if [ ! -d "${ROOT}/build_scripts/patches/${PYTORCH_SCATTER_VERSION}" ]; then
    echo "Error: patches/${PYTORCH_SCATTER_VERSION} directory does not exist"
    exit 1
fi

# Apply patches.
for patch in "${ROOT}/build_scripts/patches/${PYTORCH_SCATTER_VERSION}"/*.patch; do
    patch -p1 -d ${ROOT} -i ${patch}
done
