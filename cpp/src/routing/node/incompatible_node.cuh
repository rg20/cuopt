/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../routing_helpers.cuh"
#include "routing/fleet_info.hpp"

namespace cuopt {
namespace routing {
namespace detail {

// Default number of mutually incompatible commodity types.
// Fixed at compile time analogously to default_max_capacity_dim.
constexpr int default_n_order_types = 2;

/**
 * @brief Per-node state for the incompatible co-loading constraint.
 *
 * Prevents orders of incompatible commodity types from being simultaneously
 * loaded on the same vehicle.  Sequential service (deliver X, then pick up Y)
 * remains allowed; the constraint blocks only co-loading (X and Y aboard at
 * the same time).
 *
 * @par Data layout
 *  - Fixed / problem data (set once at model load): @c order_type, @c is_pickup
 *  - Forward data (computed by forward pass): @c fwd_count[], @c fwd_excess
 *  - Backward data (computed by backward pass): @c bwd_count[], @c bwd_excess
 *
 * @par Excess measure
 *  At any route position the "incompatibility excess" is
 *    @code excess = Σ count_i² − max_j count_j²  @endcode
 *  which equals zero iff at most one type has a positive aboard count
 *  (the feasible condition), and is strictly positive when two or more
 *  incompatible types are simultaneously in transit.
 *
 *  For two types with counts (a, b) this reduces to min(a², b²), i.e.
 *  the square of the minority count.  This is convex in the counts and
 *  guides local search toward the "greatest imbalance" state (one type
 *  dominates) which is exactly the feasible state.
 *
 * @tparam i_t         Integer type used throughout the routing solver
 * @tparam n_order_types_  Number of mutually incompatible commodity types
 */
template <typename i_t, int n_order_types_ = default_n_order_types>
class incompatible_node_t {
 public:
  static constexpr int n_order_types = n_order_types_;

  HDI incompatible_node_t() {}

  // ── Fixed (problem) data ────────────────────────────────────────────────────
  //! Commodity type index of this order node.  -1 means "no type constraint".
  int8_t order_type{-1};
  //! True if this node is a pickup (PDP); false if a delivery (PDP) or depot.
  bool is_pickup{false};
  //! True if this node is a VRP service node (pre-loaded delivery).
  //! VRP nodes are excluded from INCOMPAT co-loading accounting: they do not
  //! update fwd_count or bwd_count.  INCOMPAT enforcement only applies to PDP
  //! (pickup-delivery) routes where items are dynamically loaded/unloaded.
  bool is_vrp_node{false};

  // ── Forward data ────────────────────────────────────────────────────────────
  //! For PDP: net aboard count at END of forward fragment (pickup → +1, delivery → −1).
  //! For VRP: deliveries-made count at END of forward fragment (every delivery → +1).
  //! In both cases fwd_count[last_service_node] == bwd_count[first_service_node].
  i_t fwd_count[n_order_types] = {0};
  //! Accumulated incompatibility excess over the entire forward fragment.
  //! At each position p: excess_p = Σ fwd_count_p[i]² − max_j fwd_count_p[j]²
  //! fwd_excess = Σ_p excess_p
  i_t fwd_excess{0};

  // ── Backward data ───────────────────────────────────────────────────────────
  //! Net count of type-t orders in transit at the START of the backward
  //! fragment from this node to the terminal depot.
  //! Going backward: incremented by +1 at a type-t delivery (the pickup was
  //! before this fragment), decremented by −1 at a type-t pickup.
  i_t bwd_count[n_order_types] = {0};
  //! Accumulated incompatibility excess over the entire backward fragment.
  i_t bwd_excess{0};

  // ── Forward propagation ─────────────────────────────────────────────────────
  /**
   * @brief Propagate forward data from *this to @p next.
   *
   * @p arc_value is accepted for interface uniformity but is not used; the
   * incompatible-type constraint depends only on the nodes themselves.
   */
  void HDI calculate_forward(incompatible_node_t& next,
                             [[maybe_unused]] double arc_value = 0.) const noexcept
  {
    // Propagate accumulated forward counts.
    constexpr_for<n_order_types>([&](auto t) { next.fwd_count[t] = fwd_count[t]; });

    // Apply the next node's demand (PDP only).
    // VRP nodes do not participate in the incompatible co-loading constraint;
    // for VRP, this constraint has no well-defined semantics (all deliveries
    // are pre-loaded at the depot, so nothing is "co-loaded" dynamically).
    // PDP pickup: +1 (item goes aboard).  PDP delivery: -1 (item leaves).
    if (next.order_type >= 0 && next.order_type < n_order_types && !next.is_vrp_node) {
      i_t delta = next.is_pickup ? i_t{1} : i_t{-1};
      next.fwd_count[(int)next.order_type] += delta;
    }

    // Compute incompatibility excess at *next*:
    //   excess = Σ count_i² − max_j count_j²
    i_t sum_sq{0};
    i_t max_sq{0};
    constexpr_for<n_order_types>([&](auto t) {
      i_t sq = next.fwd_count[t] * next.fwd_count[t];
      sum_sq += sq;
      max_sq = max(max_sq, sq);
    });
    // Depot nodes (order_type < 0) do not contribute positional excess;
    // they inherit the accumulated excess unchanged to preserve the invariant
    //   fwd_excess[k] + bwd_excess[k] == constant for all k in the route.
    if (next.order_type >= 0) {
      next.fwd_excess = fwd_excess + (sum_sq - max_sq);
    } else {
      next.fwd_excess = fwd_excess;
    }
  }

  // ── Backward propagation ────────────────────────────────────────────────────
  /**
   * @brief Propagate backward data from *this (further toward end) to @p prev.
   */
  void HDI calculate_backward(incompatible_node_t& prev,
                              [[maybe_unused]] double arc_value = 0.) const noexcept
  {
    // Propagate accumulated backward counts.
    constexpr_for<n_order_types>([&](auto t) { prev.bwd_count[t] = bwd_count[t]; });

    // Apply prev node's demand in backward direction (PDP only).
    // VRP nodes do not participate — see calculate_forward for rationale.
    //  - PDP delivery at prev → one more type-t order has its pickup before prev
    //                           (it is in transit entering from the left): +1
    //  - PDP pickup at prev   → this pickup is INSIDE the backward fragment,
    //                           so subtract from the "open from the left" count: -1
    if (prev.order_type >= 0 && prev.order_type < n_order_types && !prev.is_vrp_node) {
      i_t delta = prev.is_pickup ? i_t{-1} : i_t{1};
      prev.bwd_count[(int)prev.order_type] += delta;
    }

    // Compute incompatibility excess at *prev*.
    i_t sum_sq{0};
    i_t max_sq{0};
    constexpr_for<n_order_types>([&](auto t) {
      i_t sq = prev.bwd_count[t] * prev.bwd_count[t];
      sum_sq += sq;
      max_sq = max(max_sq, sq);
    });
    // Depot nodes (order_type < 0) do not contribute positional excess;
    // they inherit the accumulated excess unchanged to preserve the invariant
    //   fwd_excess[k] + bwd_excess[k] == constant for all k in the route.
    if (prev.order_type >= 0) {
      prev.bwd_excess = bwd_excess + (sum_sq - max_sq);
    } else {
      prev.bwd_excess = bwd_excess;
    }
  }

  // ── Combine ─────────────────────────────────────────────────────────────────
  /**
   * @brief Compute the total incompatibility excess when joining a forward
   *        fragment ending at @p prev with a backward fragment starting at
   *        @p next.
   *
   * For PDP routes the invariant is:
   *   combine(prev=k, next=k+1) == fwd_excess[n_nodes]  for all split points k
   *
   * Why: fwd_excess[k] counts the violation at the pickup that introduced the
   * second type (inside the forward fragment), and bwd_excess[k+1] counts the
   * same violation again at the delivery node seen from the backward direction.
   * Their sum double-counts the violation at the boundary segment k→k+1.
   * We remove the double-count by subtracting excess(fwd_count[k]):
   *
   *   total = fwd_excess[k] + bwd_excess[k+1] - excess(fwd_count[k])
   *
   * where excess(c) = Σ c[t]² - max c[t]²  (= 0 iff single type in transit).
   *
   * @return  Total incompatibility excess (0 = fully feasible).
   */
  static i_t HDI combine(const incompatible_node_t& prev,
                         const incompatible_node_t& next,
                         [[maybe_unused]] const VehicleInfo<double>& vehicle_info,
                         [[maybe_unused]] double arc_value = 0.) noexcept
  {
    // Intra-fragment excesses.
    i_t total = prev.fwd_excess + next.bwd_excess;

    // Subtract the double-counted boundary excess.
    // prev.fwd_count[t] = items of type t in transit at the join (post-service at prev).
    // For a valid PDP route this equals next.bwd_count[t], so both fwd_excess and
    // bwd_excess already capture the same violation at the boundary segment.
    // We subtract it once to avoid double-counting.
    i_t sum_sq{0};
    i_t max_sq{0};
    constexpr_for<n_order_types>([&](auto t) {
      i_t c  = prev.fwd_count[t];
      i_t sq = c * c;
      sum_sq += sq;
      max_sq = max(max_sq, sq);
    });
    total -= (sum_sq - max_sq);

    return total;
  }

  // ── Overload for float VehicleInfo ─────────────────────────────────────────
  static i_t HDI combine(const incompatible_node_t& prev,
                         const incompatible_node_t& next,
                         [[maybe_unused]] const VehicleInfo<float>& vehicle_info,
                         [[maybe_unused]] double arc_value = 0.) noexcept
  {
    return combine(prev, next, VehicleInfo<double>{}, arc_value);
  }

  // ── Excess queries ──────────────────────────────────────────────────────────
  HDI double forward_excess([[maybe_unused]] const VehicleInfo<double>& vehicle_info) const noexcept
  {
    return static_cast<double>(fwd_excess);
  }

  HDI double forward_excess([[maybe_unused]] const VehicleInfo<float>& vehicle_info) const noexcept
  {
    return static_cast<double>(fwd_excess);
  }

  HDI double backward_excess(
    [[maybe_unused]] const VehicleInfo<double>& vehicle_info) const noexcept
  {
    return static_cast<double>(bwd_excess);
  }

  HDI double backward_excess([[maybe_unused]] const VehicleInfo<float>& vehicle_info) const noexcept
  {
    return static_cast<double>(bwd_excess);
  }

  HDI bool forward_feasible([[maybe_unused]] const VehicleInfo<double>& vehicle_info,
                            [[maybe_unused]] double weight       = 1.,
                            [[maybe_unused]] double excess_limit = 0.) const noexcept
  {
    return fwd_excess <= static_cast<i_t>(excess_limit);
  }

  HDI bool forward_feasible([[maybe_unused]] const VehicleInfo<float>& vehicle_info,
                            [[maybe_unused]] double weight       = 1.,
                            [[maybe_unused]] double excess_limit = 0.) const noexcept
  {
    return fwd_excess <= static_cast<i_t>(excess_limit);
  }

  HDI bool backward_feasible([[maybe_unused]] const VehicleInfo<double>& vehicle_info,
                             [[maybe_unused]] double weight       = 1.,
                             [[maybe_unused]] double excess_limit = 0.) const noexcept
  {
    return bwd_excess <= static_cast<i_t>(excess_limit);
  }

  HDI bool backward_feasible([[maybe_unused]] const VehicleInfo<float>& vehicle_info,
                             [[maybe_unused]] double weight       = 1.,
                             [[maybe_unused]] double excess_limit = 0.) const noexcept
  {
    return bwd_excess <= static_cast<i_t>(excess_limit);
  }

  /**
   * @brief Accumulate per-dimension infeasibility into @p inf_cost.
   *
   * Reports the full node cost = fwd_excess + bwd_excess (forward and backward
   * excess are independent; the join-point contribution is evaluated in @c combine).
   */
  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const incompatible_node_t& prev,
                    [[maybe_unused]] const VehicleInfo<double, is_device>& vehicle_info,
                    [[maybe_unused]] const incompatible_dimension_info_t& dim_info,
                    [[maybe_unused]] objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    inf_cost[dim_t::INCOMPAT] = static_cast<double>(fwd_excess) + static_cast<double>(bwd_excess);
  }

  template <bool is_device = true>
  HDI void get_cost([[maybe_unused]] const incompatible_node_t& prev,
                    [[maybe_unused]] const VehicleInfo<float, is_device>& vehicle_info,
                    [[maybe_unused]] const incompatible_dimension_info_t& dim_info,
                    [[maybe_unused]] objective_cost_t& obj_cost,
                    infeasible_cost_t& inf_cost) const noexcept
  {
    inf_cost[dim_t::INCOMPAT] = static_cast<double>(fwd_excess) + static_cast<double>(bwd_excess);
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
