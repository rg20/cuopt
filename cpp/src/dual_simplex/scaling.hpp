/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <dual_simplex/presolve.hpp>
#include <dual_simplex/simplex_solver_settings.hpp>
#include <linear_algebra/sparse_matrix.hpp>
#include <math_optimization/types.hpp>

#include <vector>

namespace cuopt::mathematical_optimization::simplex {

// objective_rescaling: whole-problem objective scalar (see barrier's Curtis-Reid +
// Pock-Chambolle branch); 1.0 (no-op) for every other scaling path. Unlike column_scaling /
// row_scaling, it cannot be folded into either of those vectors: it scales the dual y and
// reduced cost z by the same factor, independent of column_scaling's own (unrelated) effect
// on x and z. See unscale_solution below for how it's applied.
template <typename i_t, typename f_t>
i_t scaling(const lp_problem_t<i_t, f_t>& unscaled,
            const simplex_solver_settings_t<i_t, f_t>& settings,
            lp_problem_t<i_t, f_t>& scaled,
            std::vector<f_t>& column_scaling,
            std::vector<f_t>& row_scaling,
            f_t& objective_rescaling);

template <typename i_t, typename f_t>
void unscale_solution(const std::vector<f_t>& column_scaling,
                      const std::vector<f_t>& row_scaling,
                      f_t objective_rescaling,
                      const std::vector<f_t>& scaled_x,
                      const std::vector<f_t>& scaled_y,
                      const std::vector<f_t>& scaled_z,
                      std::vector<f_t>& unscaled_x,
                      std::vector<f_t>& unscaled_y,
                      std::vector<f_t>& unscaled_z);

}  // namespace cuopt::mathematical_optimization::simplex
