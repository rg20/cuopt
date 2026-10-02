#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# != 1 || ! $1 =~ ^sm_[0-9]+[a-z]?$ ]]; then
  echo "Usage: bash build_experiment.sh <CUDA architecture, e.g. sm_90>" >&2
  exit 1
fi
: "${CONDA_PREFIX:?Activate the cuOpt development conda environment first}"
experiment_source=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
experiment_repo=$(git -C "$experiment_source" rev-parse --show-toplevel)
experiment_build="$experiment_repo/cpp/build"
experiment_cccl="$experiment_build/_deps/cccl-src"
experiment_output="$experiment_build/experimental_condensed_socp"
test -f "$experiment_cccl/libcudacxx/include/cuda/stream"
mkdir -p "$experiment_output"
common=(-std=c++17 -O3 -arch="$1" -shared -Xcompiler -fPIC
  -I "$experiment_cccl/libcudacxx/include"
  -I "$experiment_cccl/thrust" -I "$experiment_cccl/cub"
  -I "$CONDA_PREFIX/include" -L "$CONDA_PREFIX/lib")

nvcc "${common[@]}" "$experiment_source/condensed_cuda.cu" \
  -lcublas -lcusolver -o "$experiment_output/condensed_cuda.so"
nvcc "${common[@]}" "$experiment_source/live_condensed.cu" \
  -lcudss -lcublas -lcusolver -ldl -o "$experiment_output/live_condensed.so"
nvcc "${common[@]}" -DCUOPT_EXPERIMENT_NO_FALLBACK "$experiment_source/live_condensed.cu" \
  -lcudss -lcublas -lcusolver -ldl -o "$experiment_output/live_condensed_nofallback.so"
nvcc "${common[@]}" "$experiment_source/capture_kkt.cpp" \
  -lcudss -lcudart -ldl -o "$experiment_output/capture_kkt.so"
echo "Experimental libraries: $experiment_output"
