// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Standalone diagnostic; allocations and input extraction belong to the replay driver.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace {
cublasHandle_t blas;
cusolverDnHandle_t solver;
constexpr int small_n = 53;
void check(int status)
{
  if (status) {
    std::fprintf(stderr, "CUDA diagnostic status %d\n", status);
    std::abort();
  }
}

__global__ void invert_blocks(const double* blocks, double* inverse, int count, int* bad)
{
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= count) return;
  double a[3][6];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 6; ++j)
      a[i][j] = j < 3 ? blocks[b * 9 + i * 3 + j] : double(j - 3 == i);
  for (int k = 0; k < 3; ++k) {
    int p = k;
    for (int i = k + 1; i < 3; ++i)
      if (fabs(a[i][k]) > fabs(a[p][k])) p = i;
    if (!isfinite(a[p][k]) || a[p][k] == 0) {
      atomicExch(bad, 1);
      return;
    }
    for (int j = 0; j < 6; ++j) {
      double tmp = a[k][j];
      a[k][j]    = a[p][j];
      a[p][j]    = tmp;
    }
    double pivot = a[k][k];
    for (int j = 0; j < 6; ++j)
      a[k][j] /= pivot;
    for (int i = 0; i < 3; ++i)
      if (i != k) {
        double multiplier = a[i][k];
        for (int j = 0; j < 6; ++j)
          a[i][j] = fma(-multiplier, a[k][j], a[i][j]);
      }
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      inverse[b * 9 + i * 3 + j] = a[i][j + 3];
}

__global__ void apply_inverse(const double* inverse, const double* coupling, double* w, int rows)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= rows * small_n) return;
  int row = t % rows, col = t / rows, block = row / 3, within = row % 3;
  double value = 0;
  for (int j = 0; j < 3; ++j)
    value = fma(inverse[block * 9 + within * 3 + j], coupling[col * rows + block * 3 + j], value);
  w[t] = value;
}

__global__ void local_rhs(const double* inverse,
                          const int* indices,
                          const int* keep,
                          const double* rhs,
                          double* local,
                          double* small,
                          int rows)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < small_n) small[row] = rhs[keep[row]];
  if (row >= rows) return;
  int block = row / 3, within = row % 3;
  double value = 0;
  for (int j = 0; j < 3; ++j) {
    int index = indices[block * 3 + j];
    if (index >= 0) value = fma(inverse[block * 9 + within * 3 + j], rhs[index], value);
  }
  local[row] = value;
}

__global__ void scatter(const int* indices,
                        const int* keep,
                        const double* local,
                        const double* small,
                        double* output,
                        int rows)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < small_n) output[keep[row]] = small[row];
  if (row < rows && indices[row] >= 0) output[indices[row]] = local[row];
}
}  // namespace

extern "C" int initialize()
{
  check(cublasCreate(&blas));
  check(cusolverDnCreate(&solver));
  int work = 0;
  check(cusolverDnDgetrf_bufferSize(solver, small_n, small_n, nullptr, small_n, &work));
  return work;
}
extern "C" void finalize()
{
  check(cusolverDnDestroy(solver));
  check(cublasDestroy(blas));
}
extern "C" void set_stream(cudaStream_t stream)
{
  check(cublasSetStream(blas, stream));
  check(cusolverDnSetStream(solver, stream));
}

extern "C" void factor(int count,
                       const double* blocks,
                       const double* coupling,
                       const double* corner,
                       double* inverse,
                       double* w,
                       double* lu,
                       double* work,
                       int* pivots,
                       int* info,
                       int* bad,
                       cudaStream_t stream)
{
  int rows = count * 3;
  check(cudaMemsetAsync(bad, 0, sizeof(int), stream));
  invert_blocks<<<(count + 127) / 128, 128, 0, stream>>>(blocks, inverse, count, bad);
  apply_inverse<<<(rows * small_n + 255) / 256, 256, 0, stream>>>(inverse, coupling, w, rows);
  check(cudaMemcpyAsync(
    lu, corner, small_n * small_n * sizeof(double), cudaMemcpyDeviceToDevice, stream));
  const double minus = -1, one = 1;
  check(cublasDgemm(blas,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    small_n,
                    small_n,
                    rows,
                    &minus,
                    coupling,
                    rows,
                    w,
                    rows,
                    &one,
                    lu,
                    small_n));
  check(cusolverDnDgetrf(solver, small_n, small_n, lu, small_n, work, pivots, info));
  check(cudaPeekAtLastError());
}

extern "C" void solve(int count,
                      const int* indices,
                      const int* keep,
                      const double* coupling,
                      const double* inverse,
                      const double* w,
                      const double* lu,
                      const int* pivots,
                      const double* rhs,
                      double* local,
                      double* small,
                      double* output,
                      int* info,
                      cudaStream_t stream)
{
  int rows = count * 3;
  local_rhs<<<(rows + 255) / 256, 256, 0, stream>>>(
    inverse, indices, keep, rhs, local, small, rows);
  const double minus = -1, one = 1;
  check(cublasDgemv(
    blas, CUBLAS_OP_T, rows, small_n, &minus, coupling, rows, local, 1, &one, small, 1));
  check(
    cusolverDnDgetrs(solver, CUBLAS_OP_N, small_n, 1, lu, small_n, pivots, small, small_n, info));
  check(cublasDgemv(blas, CUBLAS_OP_N, rows, small_n, &minus, w, rows, small, 1, &one, local, 1));
  scatter<<<(rows + 255) / 256, 256, 0, stream>>>(indices, keep, local, small, output, rows);
  check(cudaPeekAtLastError());
}
