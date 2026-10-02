// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Model-scoped live experiment. Enable only by explicitly preloading this library.
// The ordinary cuOpt refinement and convergence tests remain in the caller.
#include <cudss.h>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cub/warp/warp_reduce.cuh>
#include <map>
#include <memory>
#include <numeric>
#include <rmm/device_uvector.hpp>
#include <vector>
#include "condensed_cuda.cu"

namespace {
using execute_t = decltype(&cudssExecute);
execute_t original_execute()
{
  return reinterpret_cast<execute_t>(dlsym(RTLD_NEXT, "cudssExecute"));
}
std::map<cudssHandle_t, cudaStream_t> streams;

template <class T>
using buffer_t = rmm::device_uvector<T>;
template <class T>
void upload(buffer_t<T>& dst, const std::vector<T>& src, cudaStream_t stream)
{
  dst.resize(src.size(), stream);
  check(cudaMemcpyAsync(
    dst.data(), src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
  check(cudaStreamSynchronize(stream));
}

__global__ void gather_values(const double* values, const int* map, double* out, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = map[i] >= 0 ? values[map[i]] : double(map[i] == -2);
}
__global__ void check_solution(const int* row,
                               const int* col,
                               const double* val,
                               const double* rhs,
                               const double* x,
                               int n,
                               int* bad)
{
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  if (i >= n) return;
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  using reduce_t = cub::WarpReduce<double>;
  __shared__ reduce_t::TempStorage sums[4], magnitudes[4];
  double sum = 0, magnitude = lane == 0 ? fabs(rhs[i]) : 0;
  for (int k = row[i] + lane; k < row[i + 1]; k += 32) {
    double term = val[k] * x[col[k]];
    sum += term;
    magnitude += fabs(term);
  }
  sum       = reduce_t(sums[warp]).Sum(sum);
  magnitude = reduce_t(magnitudes[warp]).Sum(magnitude);
  if (lane == 0 &&
      (!isfinite(x[i]) || !isfinite(sum) || fabs(sum - rhs[i]) > 1e-8 * (magnitude + 1e-300)))
    atomicExch(bad, 1);
}
__global__ void warm_identity(double* blocks, double* corner, int count)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count * 9) blocks[i] = double(i % 9 / 3 == i % 3);
  if (i < 53 * 53) corner[i] = double(i / 53 == i % 53);
}

struct state_t {
  cudssHandle_t handle;
  cudssMatrix_t matrix;
  cudaStream_t stream;
  int count = 0, n = 15207;
  bool active = true;
  int factors = 0, solves = 0, fallbacks = 0;
  buffer_t<int> indices, keep, block_map, coupling_map, corner_map, pivots, info, bad;
  buffer_t<double> blocks, coupling, corner, inverse, w, lu, work, local, small, warm_rhs, warm_x;
  state_t(cudssHandle_t h, cudssMatrix_t a, cudaStream_t s)
    : handle(h),
      matrix(a),
      stream(s),
      indices(0, s),
      keep(0, s),
      block_map(0, s),
      coupling_map(0, s),
      corner_map(0, s),
      pivots(53, s),
      info(1, s),
      bad(1, s),
      blocks(0, s),
      coupling(0, s),
      corner(2809, s),
      inverse(0, s),
      w(0, s),
      lu(2809, s),
      work(0, s),
      local(0, s),
      small(53, s),
      warm_rhs(n, s),
      warm_x(n, s)
  {
  }
  ~state_t()
  {
    std::fprintf(stderr,
                 "Condensed experiment: %d factors, %d solves, %d fallbacks\n",
                 factors,
                 solves,
                 fallbacks);
    finalize();
  }
};
std::unique_ptr<state_t> state;

int root(std::vector<int>& parents, int node)
{
  while (parents[node] != node) {
    parents[node] = parents[parents[node]];
    node          = parents[node];
  }
  return node;
}
bool prepare(cudssHandle_t handle, cudssMatrix_t matrix, cudaStream_t stream)
{
  auto start = std::chrono::steady_clock::now();
  int64_t n, m, nz;
  void *row_ptr, *end_ptr, *col_ptr, *val_ptr;
  cudaDataType_t it, vt;
  cudssMatrixType_t mt;
  cudssMatrixViewType_t mv;
  cudssIndexBase_t base;
  if (cudssMatrixGetCsr(
        matrix, &n, &m, &nz, &row_ptr, &end_ptr, &col_ptr, &val_ptr, &it, &vt, &mt, &mv, &base))
    return false;
  if (n != 15207 || m != n || it != CUDA_R_32I || vt != CUDA_R_64F || mv != CUDSS_MVIEW_FULL ||
      base != CUDSS_BASE_ZERO)
    return false;
  std::vector<int> rows(n + 1), cols(nz), parents(n), position(n, -1), kept(n, -1), keep;
  check(cudaStreamSynchronize(stream));
  check(cudaMemcpy(rows.data(), row_ptr, (n + 1) * sizeof(int), cudaMemcpyDefault));
  check(cudaMemcpy(cols.data(), col_ptr, nz * sizeof(int), cudaMemcpyDefault));
  for (int i = 10102; i < 10153; ++i)
    keep.push_back(i);
  keep.push_back(n - 2);
  keep.push_back(n - 1);
  for (int i = 0; i < 53; ++i)
    kept[keep[i]] = i;
  std::iota(parents.begin(), parents.end(), 0);
  for (int i = 0; i < n; ++i)
    for (int k = rows[i]; k < rows[i + 1]; ++k) {
      int j = cols[k];
      if (j < 0 || j >= n) return false;
      if (kept[i] < 0 && kept[j] < 0) parents[root(parents, i)] = root(parents, j);
    }
  std::map<int, std::vector<int>> groups;
  for (int i = 0; i < n; ++i)
    if (kept[i] < 0) groups[root(parents, i)].push_back(i);
  if (groups.size() != 5052) return false;
  std::vector<int> indices;
  int pairs = 0, triples = 0;
  for (const auto& entry : groups) {
    const auto& group = entry.second;
    if (group.size() < 2 || group.size() > 3) return false;
    pairs += group.size() == 2;
    triples += group.size() == 3;
    for (int i : group) {
      position[i] = indices.size();
      indices.push_back(i);
    }
    if (group.size() == 2) indices.push_back(-1);
  }
  if (pairs != 2 || triples != 5050) return false;
  int count = groups.size(), padded = count * 3;
  std::vector<int> block_map(count * 9, -1), coupling_map(padded * 53, -1), corner_map(2809, -1);
  for (int i = 0; i < padded; ++i)
    if (indices[i] < 0) block_map[i / 3 * 9 + 8] = -2;
  for (int i = 0; i < n; ++i)
    for (int k = rows[i]; k < rows[i + 1]; ++k) {
      int j = cols[k], target;
      if (kept[i] >= 0 && kept[j] >= 0) {
        target = kept[j] * 53 + kept[i];
        if (corner_map[target] != -1) return false;
        corner_map[target] = k;
      } else if (kept[i] < 0 && kept[j] >= 0) {
        target = kept[j] * padded + position[i];
        if (coupling_map[target] != -1) return false;
        coupling_map[target] = k;
      } else if (kept[i] < 0 && kept[j] < 0) {
        if (position[i] / 3 != position[j] / 3) return false;
        target = position[i] / 3 * 9 + position[i] % 3 * 3 + position[j] % 3;
        if (block_map[target] != -1) return false;
        block_map[target] = k;
      }
    }
  auto maps_done      = std::chrono::steady_clock::now();
  int workspace       = initialize();
  auto libraries_done = std::chrono::steady_clock::now();
  state               = std::make_unique<state_t>(handle, matrix, stream);
  auto& s             = *state;
  s.count             = count;
  set_stream(stream);
  upload(s.indices, indices, stream);
  upload(s.keep, keep, stream);
  upload(s.block_map, block_map, stream);
  upload(s.coupling_map, coupling_map, stream);
  upload(s.corner_map, corner_map, stream);
  s.blocks.resize(count * 9, stream);
  s.inverse.resize(count * 9, stream);
  s.coupling.resize(padded * 53, stream);
  s.w.resize(padded * 53, stream);
  s.local.resize(padded, stream);
  s.work.resize(workspace, stream);
  check(cudaMemsetAsync(s.coupling.data(), 0, s.coupling.size() * sizeof(double), stream));
  check(cudaMemsetAsync(s.warm_rhs.data(), 0, n * sizeof(double), stream));
  check(cudaStreamSynchronize(stream));
  auto allocation_done = std::chrono::steady_clock::now();
  warm_identity<<<(count * 9 + 255) / 256, 256, 0, stream>>>(
    s.blocks.data(), s.corner.data(), count);
  factor(count,
         s.blocks.data(),
         s.coupling.data(),
         s.corner.data(),
         s.inverse.data(),
         s.w.data(),
         s.lu.data(),
         s.work.data(),
         s.pivots.data(),
         s.info.data(),
         s.bad.data(),
         stream);
  solve(count,
        s.indices.data(),
        s.keep.data(),
        s.coupling.data(),
        s.inverse.data(),
        s.w.data(),
        s.lu.data(),
        s.pivots.data(),
        s.warm_rhs.data(),
        s.local.data(),
        s.small.data(),
        s.warm_x.data(),
        s.info.data(),
        stream);
  check(cudaStreamSynchronize(stream));
  auto warm_done = std::chrono::steady_clock::now();
  std::fprintf(stderr,
               "Condensed setup ms: maps=%.3f libraries=%.3f buffers=%.3f warmup=%.3f\n",
               std::chrono::duration<double, std::milli>(maps_done - start).count(),
               std::chrono::duration<double, std::milli>(libraries_done - maps_done).count(),
               std::chrono::duration<double, std::milli>(allocation_done - libraries_done).count(),
               std::chrono::duration<double, std::milli>(warm_done - allocation_done).count());
  std::fprintf(
    stderr,
    "Condensed experiment enabled: 53 retained, %d local blocks; unchanged outer refinement\n",
    count);
  return true;
}
}  // namespace

extern "C" cudssStatus_t cudssSetStream(cudssHandle_t handle, cudaStream_t stream)
{
  auto real       = reinterpret_cast<decltype(&cudssSetStream)>(dlsym(RTLD_NEXT, "cudssSetStream"));
  streams[handle] = stream;
  return real(handle, stream);
}
extern "C" cudssStatus_t cudssDestroy(cudssHandle_t handle)
{
  auto real = reinterpret_cast<decltype(&cudssDestroy)>(dlsym(RTLD_NEXT, "cudssDestroy"));
  if (state && state->handle == handle) state.reset();
  streams.erase(handle);
  return real(handle);
}
#ifdef CUOPT_EXPERIMENT_NO_FALLBACK
extern "C" cudssStatus_t cudssDataGet(
  cudssHandle_t h, cudssData_t d, cudssDataParam_t param, void* value, size_t size, size_t* written)
{
  if (state && state->handle == h) {
    // No sparse factors exist. Factor/solve return status already checks dense LU.
    size_t bytes = param == CUDSS_DATA_LU_NNZ ? sizeof(int64_t) : sizeof(int);
    if ((param != CUDSS_DATA_LU_NNZ && param != CUDSS_DATA_INFO) || size < bytes)
      return CUDSS_STATUS_INVALID_VALUE;
    std::memset(value, 0, bytes);
    if (written) *written = bytes;
    return CUDSS_STATUS_SUCCESS;
  }
  auto real = reinterpret_cast<decltype(&cudssDataGet)>(dlsym(RTLD_NEXT, "cudssDataGet"));
  return real(h, d, param, value, size, written);
}
#endif
extern "C" cudssStatus_t cudssExecute(cudssHandle_t h,
                                      int phase,
                                      cudssConfig_t c,
                                      cudssData_t d,
                                      cudssMatrix_t a,
                                      cudssMatrix_t x,
                                      cudssMatrix_t b)
{
  auto real = original_execute();
#ifdef CUOPT_EXPERIMENT_NO_FALLBACK
  if (phase == CUDSS_PHASE_REORDERING) {
    if (state || !prepare(h, a, streams[h])) {
      std::fprintf(stderr,
                   "No-fallback experiment: unsupported matrix; refusing cuDSS execution\n");
      return CUDSS_STATUS_INVALID_VALUE;
    }
    std::fprintf(stderr,
                 "No-fallback experiment: cuDSS ordering and symbolic phases skipped; sparse "
                 "factor count is zero\n");
    return CUDSS_STATUS_SUCCESS;
  }
  if (phase == CUDSS_PHASE_SYMBOLIC_FACTORIZATION && state && state->handle == h &&
      state->matrix == a)
    return CUDSS_STATUS_SUCCESS;
#else
  if (phase == CUDSS_PHASE_SYMBOLIC_FACTORIZATION) {
    auto status = real(h, phase, c, d, a, x, b);
    if (status == CUDSS_STATUS_SUCCESS && !state) prepare(h, a, streams[h]);
    return status;
  }
#endif
  if (!state || state->handle != h || state->matrix != a || !state->active ||
      (phase != CUDSS_PHASE_FACTORIZATION && phase != CUDSS_PHASE_SOLVE)) {
#ifdef CUOPT_EXPERIMENT_NO_FALLBACK
    return CUDSS_STATUS_INVALID_VALUE;
#else
    return real(h, phase, c, d, a, x, b);
#endif
  }
  auto& s = *state;
  int64_t n, m, nz;
  void *rp, *ep, *cp, *vp;
  cudaDataType_t it, vt;
  cudssMatrixType_t mt;
  cudssMatrixViewType_t mv;
  cudssIndexBase_t base;
  check(cudssMatrixGetCsr(a, &n, &m, &nz, &rp, &ep, &cp, &vp, &it, &vt, &mt, &mv, &base));
  if (phase == CUDSS_PHASE_FACTORIZATION) {
    ++s.factors;
    gather_values<<<(s.blocks.size() + 255) / 256, 256, 0, s.stream>>>(
      (double*)vp, s.block_map.data(), s.blocks.data(), s.blocks.size());
    gather_values<<<(s.coupling.size() + 255) / 256, 256, 0, s.stream>>>(
      (double*)vp, s.coupling_map.data(), s.coupling.data(), s.coupling.size());
    gather_values<<<11, 256, 0, s.stream>>>(
      (double*)vp, s.corner_map.data(), s.corner.data(), 2809);
    factor(s.count,
           s.blocks.data(),
           s.coupling.data(),
           s.corner.data(),
           s.inverse.data(),
           s.w.data(),
           s.lu.data(),
           s.work.data(),
           s.pivots.data(),
           s.info.data(),
           s.bad.data(),
           s.stream);
    int info = 0, bad = 0;
    check(cudaMemcpyAsync(&info, s.info.data(), sizeof(int), cudaMemcpyDeviceToHost, s.stream));
    check(cudaMemcpyAsync(&bad, s.bad.data(), sizeof(int), cudaMemcpyDeviceToHost, s.stream));
    check(cudaStreamSynchronize(s.stream));
    const char* fail_at = std::getenv("CUOPT_CONDENSED_TEST_FAIL_FACTOR");
    if (fail_at && s.factors == std::atoi(fail_at)) bad = 1;
    if (!info && !bad) return CUDSS_STATUS_SUCCESS;
  } else {
    ++s.solves;
    int64_t nr, nc, ld;
    void *rhs, *out;
    cudssLayout_t layout;
    check(cudssMatrixGetDn(b, &nr, &nc, &ld, &rhs, &vt, &layout));
    check(nr == s.n && nc == 1 ? 0 : 1);
    check(cudssMatrixGetDn(x, &nr, &nc, &ld, &out, &vt, &layout));
    solve(s.count,
          s.indices.data(),
          s.keep.data(),
          s.coupling.data(),
          s.inverse.data(),
          s.w.data(),
          s.lu.data(),
          s.pivots.data(),
          (double*)rhs,
          s.local.data(),
          s.small.data(),
          (double*)out,
          s.info.data(),
          s.stream);
    check_solution<<<(s.n + 3) / 4, 128, 0, s.stream>>>(
      (int*)rp, (int*)cp, (double*)vp, (double*)rhs, (double*)out, s.n, s.bad.data());
    int info = 0, bad = 0;
    check(cudaMemcpyAsync(&info, s.info.data(), sizeof(int), cudaMemcpyDeviceToHost, s.stream));
    check(cudaMemcpyAsync(&bad, s.bad.data(), sizeof(int), cudaMemcpyDeviceToHost, s.stream));
    check(cudaStreamSynchronize(s.stream));
    if (!info && !bad) return CUDSS_STATUS_SUCCESS;
  }
  s.active = false;
#ifdef CUOPT_EXPERIMENT_NO_FALLBACK
  std::fprintf(stderr, "No-fallback experiment: numerical check failed; stopping without cuDSS\n");
  return CUDSS_STATUS_INTERNAL_ERROR;
#else
  ++s.fallbacks;
  std::fprintf(stderr, "Condensed experiment reverting to cuDSS after numerical check\n");
  auto status = real(h, CUDSS_PHASE_FACTORIZATION, c, d, a, x, b);
  if (status != CUDSS_STATUS_SUCCESS || phase == CUDSS_PHASE_FACTORIZATION) return status;
  return real(h, phase, c, d, a, x, b);
#endif
}
