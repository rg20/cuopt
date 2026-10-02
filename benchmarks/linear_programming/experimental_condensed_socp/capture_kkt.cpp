// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>
#include <cudss.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

// Diagnostic-only capture. Synchronization makes these runs unsuitable for timing.
namespace {
int factor = -1;
int solve  = 0;
void check(bool ok)
{
  if (!ok) std::abort();
}
void write_device(FILE* f, const void* ptr, size_t bytes)
{
  std::vector<unsigned char> host(bytes);
  check(cudaMemcpy(host.data(), ptr, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
  check(std::fwrite(host.data(), 1, bytes, f) == bytes);
}
FILE* open_dump(const char* suffix)
{
  char name[1024];
  std::snprintf(name,
                sizeof(name),
                "%s/f%03d-s%03d-%s.bin",
                std::getenv("CUOPT_CAPTURE_KKT"),
                factor,
                solve,
                suffix);
  FILE* f = std::fopen(name, "wb");
  check(f != nullptr);
  return f;
}
void capture_matrix(cudssMatrix_t a)
{
  int64_t nr, nc, nz;
  void *row, *end, *col, *val;
  cudaDataType_t it, vt;
  cudssMatrixType_t mt;
  cudssMatrixViewType_t mv;
  cudssIndexBase_t base;
  check(cudssMatrixGetCsr(a, &nr, &nc, &nz, &row, &end, &col, &val, &it, &vt, &mt, &mv, &base) ==
        CUDSS_STATUS_SUCCESS);
  check(it == CUDA_R_32I && vt == CUDA_R_64F && base == CUDSS_BASE_ZERO);
  FILE* f        = open_dump("matrix");
  int64_t meta[] = {nr, nc, nz, int64_t(mt), int64_t(mv)};
  check(std::fwrite(meta, sizeof(meta), 1, f) == 1);
  write_device(f, row, (nr + 1) * sizeof(int32_t));
  write_device(f, col, nz * sizeof(int32_t));
  write_device(f, val, nz * sizeof(double));
  check(std::fclose(f) == 0);
}
void capture_vector(cudssMatrix_t a, const char* suffix)
{
  int64_t nr, nc, ld;
  void* val;
  cudaDataType_t vt;
  cudssLayout_t layout;
  check(cudssMatrixGetDn(a, &nr, &nc, &ld, &val, &vt, &layout) == CUDSS_STATUS_SUCCESS);
  check(vt == CUDA_R_64F && nc == 1);
  FILE* f = open_dump(suffix);
  write_device(f, val, nr * sizeof(double));
  check(std::fclose(f) == 0);
}
}  // namespace
extern "C" cudssStatus_t cudssExecute(cudssHandle_t h,
                                      int phase,
                                      cudssConfig_t c,
                                      cudssData_t d,
                                      cudssMatrix_t a,
                                      cudssMatrix_t x,
                                      cudssMatrix_t b)
{
  using fn_t = decltype(&cudssExecute);
  auto real  = reinterpret_cast<fn_t>(dlsym(RTLD_NEXT, "cudssExecute"));
  check(real != nullptr);
  const bool capture = std::getenv("CUOPT_CAPTURE_KKT") != nullptr;
  if (capture && phase == CUDSS_PHASE_FACTORIZATION) {
    check(cudaDeviceSynchronize() == cudaSuccess);
    ++factor;
    solve = 0;
    capture_matrix(a);
  }
  if (capture && phase == CUDSS_PHASE_SOLVE) {
    check(cudaDeviceSynchronize() == cudaSuccess);
    capture_vector(b, "rhs");
  }
  auto result = real(h, phase, c, d, a, x, b);
  if (capture && phase == CUDSS_PHASE_SOLVE && result == CUDSS_STATUS_SUCCESS) {
    check(cudaDeviceSynchronize() == cudaSuccess);
    capture_vector(x, "solution");
    ++solve;
  }
  return result;
}
