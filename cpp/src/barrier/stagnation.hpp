/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace cuopt::mathematical_optimization::barrier {

// Observe relative primal, dual, complementarity and objective-gap residuals.
// A patience interval is meaningful only while the same single criterion is unmet.
template <typename f_t>
class stagnation_detector_t {
 public:
  bool update(const std::array<f_t, 4>& residuals,
              const std::array<f_t, 4>& tolerances,
              const std::array<f_t, 4>& acceptance_limits,
              bool objective_gap_active)
  {
    constexpr std::size_t patience = 30;
    constexpr f_t progress_factor  = 0.95;
    constexpr f_t recovery_factor  = 1.25;
    const std::size_t active_count = objective_gap_active ? 4 : 3;
    std::size_t candidate          = no_candidate;
    for (std::size_t i = 0; i < active_count; ++i) {
      if (!std::isfinite(residuals[i]) || residuals[i] < 0) { return reset(); }
      if (residuals[i] < tolerances[i]) { continue; }
      if (candidate != no_candidate || residuals[i] >= acceptance_limits[i]) { return reset(); }
      candidate = i;
    }
    if (candidate == no_candidate) { return reset(); }

    const f_t residual = residuals[candidate];
    if (candidate != candidate_) {
      candidate_ = candidate;
      best_ = progress_reference_  = residual;
      iterations_without_progress_ = 0;
      return false;
    }
    best_ = std::min(best_, residual);
    // Keep the reference until the *cumulative* improvement is meaningful. Small
    // improvements must add up; temporary increases must not reset the reference.
    if (best_ <= progress_factor * progress_reference_) {
      progress_reference_          = best_;
      iterations_without_progress_ = 0;
    } else if (iterations_without_progress_ < patience) {
      ++iterations_without_progress_;
    }
    // A stalled best value alone does not justify accepting a deteriorated iterate.
    return iterations_without_progress_ >= patience && residual <= recovery_factor * best_;
  }

 private:
  bool reset()
  {
    candidate_                   = no_candidate;
    iterations_without_progress_ = 0;
    return false;
  }

  static constexpr std::size_t no_candidate = 4;
  std::size_t candidate_                    = no_candidate;
  std::size_t iterations_without_progress_  = 0;
  f_t best_                                 = 0;
  f_t progress_reference_                   = 0;
};

}  // namespace cuopt::mathematical_optimization::barrier
