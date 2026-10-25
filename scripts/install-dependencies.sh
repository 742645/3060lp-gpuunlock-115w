#!/bin/bash
# Install the user-approved driver, CUDA/compiler and DKMS packaging dependencies.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    nvidia-driver-580 nvidia-cuda-toolkit build-essential dkms \
    debhelper dh-dkms zstd gcc-12 g++-12
dpkg-query -W nvidia-driver-580 nvidia-cuda-toolkit build-essential dkms dh-dkms
