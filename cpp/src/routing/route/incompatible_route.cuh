/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../node/incompatible_node.cuh"
#include "../routing_helpers.cuh"
#include "../solution/solution_handle.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>

#include <rmm/device_uvector.hpp>

namespace cuopt {
namespace routing {
namespace detail {

/**
 * @brief Per-route storage for the incompatible co-loading dimension.
 *
 * Mirrors the structure of @c capacity_route_t but tailored for tracking
 * commodity-type counts rather than scalar loads.
 *
 * Storage layout (all arrays use the same @c stride = n_nodes_route + 1):
 *
 *  Fixed data (written once at route initialisation):
 *    order_type_data[stride]       – commodity type per node position (-1, 0, 1, …)
 *    is_pickup_data[stride]        – 0 = depot/PDP-delivery, 1 = PDP-pickup, 2 = VRP-service-node
 *
 *  Forward data (updated by forward pass):
 *    fwd_count[n_order_types × stride]  – net type-t count aboard, row-major
 *    fwd_excess[stride]                 – accumulated incompatibility excess
 *
 *  Backward data (updated by backward pass):
 *    bwd_count[n_order_types × stride]
 *    bwd_excess[stride]
 *
 * @tparam i_t  Integer type used throughout the routing solver.
 * @tparam f_t  Floating-point type (unused here; kept for interface uniformity).
 */
template <typename i_t, typename f_t>
class incompatible_route_t {
 public:
  // Compile-time number of order types (same default as incompatible_node_t).
  static constexpr int n_order_types = default_n_order_types;

  using node_t = incompatible_node_t<i_t, n_order_types>;

  incompatible_route_t(solution_handle_t<i_t, f_t> const* sol_handle_,
                       incompatible_dimension_info_t dim_info_)
    : dim_info(dim_info_),
      order_type_data(0, sol_handle_->get_stream()),
      is_pickup_data(0, sol_handle_->get_stream()),
      fwd_count(0, sol_handle_->get_stream()),
      fwd_excess(0, sol_handle_->get_stream()),
      bwd_count(0, sol_handle_->get_stream()),
      bwd_excess(0, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("zero incompatible_route_t ctor");
  }

  incompatible_route_t(const incompatible_route_t& other,
                       solution_handle_t<i_t, f_t> const* sol_handle_)
    : dim_info(other.dim_info),
      order_type_data(other.order_type_data, sol_handle_->get_stream()),
      is_pickup_data(other.is_pickup_data, sol_handle_->get_stream()),
      fwd_count(other.fwd_count, sol_handle_->get_stream()),
      fwd_excess(other.fwd_excess, sol_handle_->get_stream()),
      bwd_count(other.bwd_count, sol_handle_->get_stream()),
      bwd_excess(other.bwd_excess, sol_handle_->get_stream())
  {
    raft::common::nvtx::range fun_scope("incompatible_route_t copy ctor");
  }

  incompatible_route_t& operator=(incompatible_route_t&&) = default;

  void resize(i_t max_nodes_per_route, rmm::cuda_stream_view stream)
  {
    order_type_data.resize(max_nodes_per_route, stream);
    is_pickup_data.resize(max_nodes_per_route, stream);
    fwd_count.resize(n_order_types * max_nodes_per_route, stream);
    fwd_excess.resize(max_nodes_per_route, stream);
    bwd_count.resize(n_order_types * max_nodes_per_route, stream);
    bwd_excess.resize(max_nodes_per_route, stream);
  }

  // ── Device-side view ───────────────────────────────────────────────────────
  struct view_t {
    bool is_empty() const { return fwd_excess.empty(); }

    /**
     * @brief Reconstruct a full incompatible_node_t from the stored arrays at
     *        route position @p idx.
     */
    DI node_t get_node(i_t idx) const
    {
      node_t n;
      n.order_type  = static_cast<int8_t>(order_type_data[idx]);
      n.is_pickup   = (is_pickup_data[idx] == i_t{1});
      n.is_vrp_node = (is_pickup_data[idx] == i_t{2});
      constexpr_for<n_order_types>([&](auto t) {
        n.fwd_count[t] = fwd_count[t * stride + idx];
        n.bwd_count[t] = bwd_count[t * stride + idx];
      });
      n.fwd_excess = fwd_excess[idx];
      n.bwd_excess = bwd_excess[idx];
      return n;
    }

    DI void set_node(i_t idx, const node_t& node)
    {
      order_type_data[idx] = static_cast<i_t>(node.order_type);
      // Encoding: 0 = depot/PDP-delivery, 1 = PDP-pickup, 2 = VRP-service-node
      is_pickup_data[idx] = node.is_vrp_node ? i_t{2} : (node.is_pickup ? i_t{1} : i_t{0});
      set_forward_data(idx, node);
      set_backward_data(idx, node);
    }

    DI void set_forward_data(i_t idx, const node_t& node)
    {
      constexpr_for<n_order_types>(
        [&](auto t) { fwd_count[t * stride + idx] = node.fwd_count[t]; });
      fwd_excess[idx] = node.fwd_excess;
    }

    DI void set_backward_data(i_t idx, const node_t& node)
    {
      constexpr_for<n_order_types>(
        [&](auto t) { bwd_count[t * stride + idx] = node.bwd_count[t]; });
      bwd_excess[idx] = node.bwd_excess;
    }

    DI void copy_forward_data(const view_t& orig, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      constexpr_for<n_order_types>([&](auto t) {
        block_copy(fwd_count.subspan(t * stride + write_start),
                   orig.fwd_count.subspan(t * orig.stride + start_idx),
                   size);
      });
      block_copy(fwd_excess.subspan(write_start), orig.fwd_excess.subspan(start_idx), size);
    }

    DI void copy_backward_data(const view_t& orig, i_t start_idx, i_t end_idx, i_t write_start)
    {
      i_t size = end_idx - start_idx;
      constexpr_for<n_order_types>([&](auto t) {
        block_copy(bwd_count.subspan(t * stride + write_start),
                   orig.bwd_count.subspan(t * orig.stride + start_idx),
                   size);
      });
      block_copy(bwd_excess.subspan(write_start), orig.bwd_excess.subspan(start_idx), size);
    }

    DI void copy_fixed_route_data(const view_t& orig, i_t from_idx, i_t to_idx, i_t write_start)
    {
      i_t size = to_idx - from_idx;
      block_copy(
        order_type_data.subspan(write_start), orig.order_type_data.subspan(from_idx), size);
      block_copy(is_pickup_data.subspan(write_start), orig.is_pickup_data.subspan(from_idx), size);
    }

    DI void compute_cost([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                         const i_t n_nodes,
                         [[maybe_unused]] objective_cost_t& obj_cost,
                         infeasible_cost_t& inf_cost) const noexcept
    {
      // Use only the forward excess at the last node (which equals fwd_excess
      // at the return depot after propagation through all PDP nodes).
      // VRP nodes do not contribute to fwd_excess (see incompatible_node_t),
      // so this is always 0 for VRP routes.
      // cost_combine propagates forward to a fresh depot (bwd_excess=0) and
      // calls get_cost, which returns fwd_excess + 0 = fwd_excess[n_nodes].
      // This keeps compute_cost consistent with cost_combine so that
      // total_delta == cost_after - cost_before in the OX coherence check.
      inf_cost[dim_t::INCOMPAT] = static_cast<double>(fwd_excess[n_nodes]);
    }

    /**
     * @brief Allocate a shared-memory view for a route of @p n_nodes_route nodes.
     */
    static DI thrust::tuple<view_t, i_t*> create_shared_route(
      i_t* shmem, const incompatible_dimension_info_t /*dim_info*/, i_t n_nodes_route)
    {
      view_t v;
      v.stride    = n_nodes_route + 1;
      i_t* sh_ptr = shmem;

      size_t sz_single = static_cast<size_t>(v.stride);
      size_t sz_typed  = static_cast<size_t>(v.stride * n_order_types);

      thrust::tie(v.order_type_data, sh_ptr) = wrap_ptr_as_span<i_t>(sh_ptr, sz_single);
      thrust::tie(v.is_pickup_data, sh_ptr)  = wrap_ptr_as_span<i_t>(sh_ptr, sz_single);
      thrust::tie(v.fwd_count, sh_ptr)       = wrap_ptr_as_span<i_t>(sh_ptr, sz_typed);
      thrust::tie(v.fwd_excess, sh_ptr)      = wrap_ptr_as_span<i_t>(sh_ptr, sz_single);
      thrust::tie(v.bwd_count, sh_ptr)       = wrap_ptr_as_span<i_t>(sh_ptr, sz_typed);
      thrust::tie(v.bwd_excess, sh_ptr)      = wrap_ptr_as_span<i_t>(sh_ptr, sz_single);

      return thrust::make_tuple(v, sh_ptr);
    }

    // Fixed data
    raft::device_span<i_t> order_type_data;  // int8_t stored as i_t
    raft::device_span<i_t> is_pickup_data;   // bool stored as i_t

    // Forward data
    raft::device_span<i_t> fwd_count;   // [n_order_types * stride], row-major
    raft::device_span<i_t> fwd_excess;  // [stride]

    // Backward data
    raft::device_span<i_t> bwd_count;   // [n_order_types * stride], row-major
    raft::device_span<i_t> bwd_excess;  // [stride]

    incompatible_dimension_info_t dim_info;
    i_t stride{0};
  };

  view_t view()
  {
    view_t v;
    v.dim_info = dim_info;
    // stride is determined from the allocated size divided by n_order_types
    i_t sz            = static_cast<i_t>(fwd_excess.size());
    v.stride          = sz;
    v.order_type_data = raft::device_span<i_t>{order_type_data.data(), order_type_data.size()};
    v.is_pickup_data  = raft::device_span<i_t>{is_pickup_data.data(), is_pickup_data.size()};
    v.fwd_count       = raft::device_span<i_t>{fwd_count.data(), fwd_count.size()};
    v.fwd_excess      = raft::device_span<i_t>{fwd_excess.data(), fwd_excess.size()};
    v.bwd_count       = raft::device_span<i_t>{bwd_count.data(), bwd_count.size()};
    v.bwd_excess      = raft::device_span<i_t>{bwd_excess.data(), bwd_excess.size()};
    return v;
  }

  /**
   * @brief Shared-memory footprint for a route of @p route_size positions.
   *
   * Arrays: order_type, is_pickup, fwd_count (×n_types), fwd_excess,
   *         bwd_count (×n_types), bwd_excess.
   */
  HDI static size_t get_shared_size(i_t route_size,
                                    [[maybe_unused]] incompatible_dimension_info_t /*dim_info*/)
  {
    // 2 fixed arrays + 2*(n_types + 1) computed arrays, each of length route_size
    return static_cast<size_t>(2 + 2 * (n_order_types + 1)) * route_size * sizeof(i_t);
  }

  // ── Host-side storage ──────────────────────────────────────────────────────
  incompatible_dimension_info_t dim_info;

  rmm::device_uvector<i_t> order_type_data;  //!< fixed; int8_t encoded as i_t
  rmm::device_uvector<i_t> is_pickup_data;   //!< fixed; bool encoded as i_t

  rmm::device_uvector<i_t> fwd_count;   //!< [n_order_types * n_nodes]
  rmm::device_uvector<i_t> fwd_excess;  //!< [n_nodes]

  rmm::device_uvector<i_t> bwd_count;   //!< [n_order_types * n_nodes]
  rmm::device_uvector<i_t> bwd_excess;  //!< [n_nodes]
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
