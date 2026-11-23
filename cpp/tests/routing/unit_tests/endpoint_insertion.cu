/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/solver_settings.hpp>
#include <routing/problem/problem.cuh>
#include <routing/solution/solution.cuh>
#include <routing/solution/solution_handle.cuh>
#include <utilities/copy_helpers.hpp>

#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace cuopt::routing::test {

TEST(EndpointInsertion, ScoresCompletePrefixBeforeSuffix)
{
  // Two requests on a line: P_A=0, D_A=1, P_B=10, D_B=11.
  // Location 0 is a depot with zero departure and return costs.
  // clang-format off
  const std::vector<float> h_cost_matrix{
    0,  0,  0,  0,  0,
    0,  0, 10,  1, 11,
    0, 10,  0,  9,  1,
    0,  1,  9,  0, 10,
    0, 11,  1, 10,  0};
  // clang-format on
  const std::vector<int> h_order_locations{1, 2, 3, 4};
  const std::vector<int> h_pickups{0, 1};
  const std::vector<int> h_deliveries{2, 3};

  raft::handle_t handle;
  auto stream            = handle.get_stream();
  auto d_cost_matrix     = device_copy(h_cost_matrix, stream);
  auto d_order_locations = device_copy(h_order_locations, stream);
  auto d_pickups         = device_copy(h_pickups, stream);
  auto d_deliveries      = device_copy(h_deliveries, stream);

  data_model_view_t<int, float> data_model(&handle, 5, 1, 4);
  data_model.add_cost_matrix(d_cost_matrix.data());
  data_model.set_order_locations(d_order_locations.data());
  data_model.set_pickup_delivery_pairs(d_pickups.data(), d_deliveries.data());

  solver_settings_t<int, float> settings;
  detail::problem_t<int, float> problem(data_model, settings);
  detail::solution_handle_t<int, float> sol_handle(stream);
  detail::solution_t<int, float, request_t::PDP> solution(problem, 0, &sol_handle);
  const std::vector<detail::NodeInfo<>> partial_route{
    {0, 1, node_type_t::PICKUP}, {1, 2, node_type_t::PICKUP}, {3, 4, node_type_t::DELIVERY}};
  solution.add_routes({{0, partial_route}});
  solution.compute_backward_forward();
  solution.compute_cost();
  handle.sync_stream();

  const detail::infeasible_cost_t weights;
  const detail::NodeInfo<> missing_delivery{2, 3, node_type_t::DELIVERY};
  ASSERT_EQ(solution.get_n_routes(), 1);
  ASSERT_EQ(solution.get_route(0).n_nodes.value(stream), 4);
  ASSERT_DOUBLE_EQ(solution.get_total_cost(weights), 11.);
  ASSERT_EQ(solution.route_node_map.get_route_id(partial_route.front()), 0);
  ASSERT_EQ(solution.route_node_map.get_route_id(missing_delivery), -1);

  // Repair D_A in P_A,P_B,D_B. The three legal insertions cost 11, 29, 21.
  // Scoring only D_A and the suffix incorrectly favors appending D_A.
  solution.add_nodes_to_best({missing_delivery}, weights);
  solution.compute_cost();
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  handle.sync_stream();

  ASSERT_EQ(solution.get_n_routes(), 1);
  const auto& route = solution.get_route(0);
  ASSERT_EQ(route.n_nodes.value(stream), 5);
  EXPECT_TRUE(solution.is_feasible());
  EXPECT_DOUBLE_EQ(solution.get_total_cost(weights), 11.);

  const auto h_nodes = host_copy(route.dimensions.requests.node_info.data() + 1, 4, stream);
  const std::vector<int> expected_orders{0, 2, 1, 3};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(h_nodes[i].node(), expected_orders[i]) << "position " << i;
  }
}

}  // namespace cuopt::routing::test
