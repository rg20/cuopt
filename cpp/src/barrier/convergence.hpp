/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>

namespace cuopt::mathematical_optimization::barrier {

template <typename i_t, typename f_t>
struct convergence_metrics_t {
  i_t iteration;
  f_t primal_objective;
  f_t dual_objective;
  // Primal infeasibility, dual infeasibility, complementarity, objective gap.
  std::array<f_t, 4> absolute;
  std::array<f_t, 4> relative;
};

struct convergence_observation_t {
  bool valid           = false;
  bool converged       = false;
  bool save_checkpoint = false;
  bool stagnated       = false;
};

// Quality belongs to a real iterate; progress may occur in different criteria
// at different iterations. Never use the componentwise envelope as a solution.
template <typename i_t, typename f_t>
class convergence_monitor_t {
 public:
  convergence_monitor_t(const std::array<f_t, 4>& tolerances,
                        const std::array<f_t, 4>& acceptance_limits,
                        bool objective_gap_active)
    : tolerances_(tolerances),
      acceptance_limits_(acceptance_limits),
      active_count_(objective_gap_active ? 4 : 3)
  {
  }

  convergence_observation_t observe(const convergence_metrics_t<i_t, f_t>& metrics)
  {
    convergence_observation_t result;
    if (!std::isfinite(metrics.primal_objective) || !std::isfinite(metrics.dual_objective)) {
      return result;
    }
    std::array<f_t, 4> normalized{};
    f_t score       = 0;
    bool acceptable = true;
    for (std::size_t i = 0; i < active_count_; ++i) {
      if (!std::isfinite(tolerances_[i]) || tolerances_[i] <= 0 ||
          !std::isfinite(acceptance_limits_[i]) || acceptance_limits_[i] <= 0 ||
          !std::isfinite(metrics.relative[i]) || metrics.relative[i] < 0 ||
          !std::isfinite(metrics.absolute[i]) || metrics.absolute[i] < 0) {
        return result;
      }
      normalized[i] = metrics.relative[i] / tolerances_[i];
      if (!std::isfinite(normalized[i])) { return result; }
      score      = std::max(score, normalized[i]);
      acceptable = acceptable && metrics.relative[i] < acceptance_limits_[i];
    }
    result.valid                = true;
    result.converged            = score < 1;
    const bool first_checkpoint = !checkpoint_.has_value();
    if (acceptable && score < best_score_) {
      checkpoint_            = metrics;
      best_score_            = score;
      result.save_checkpoint = true;
    }
    if (!checkpoint_) { return result; }

    if (result.converged) { return result; }
    for (std::size_t i = 0; i < active_count_; ++i) {
      // Improvements below a satisfied tolerance do not prolong the solve.
      const f_t value = std::max(f_t{1}, normalized[i]);
      envelope_[i]    = first_checkpoint ? value : std::min(envelope_[i], value);
    }

    // Compare historical best residuals across a fixed rolling window. Local
    // rebounds above those best values cannot repeatedly renew a patience timer.
    // The componentwise envelope is only progress evidence, never a solution.
    if (history_count_ == progress_window) {
      bool progress        = false;
      const auto& previous = history_[history_next_];
      for (std::size_t i = 0; i < active_count_; ++i) {
        const bool newly_satisfied = previous[i] > 1 && envelope_[i] == 1;
        progress = progress || newly_satisfied || envelope_[i] <= progress_factor * previous[i];
      }
      result.stagnated = !progress;
    } else {
      ++history_count_;
    }
    history_[history_next_] = envelope_;
    history_next_           = (history_next_ + 1) % history_.size();
    return result;
  }

  const std::optional<convergence_metrics_t<i_t, f_t>>& checkpoint() const { return checkpoint_; }
  f_t best_score() const { return best_score_; }

 private:
  static constexpr std::size_t progress_window = 30;
  static constexpr f_t progress_factor         = 0.95;
  const std::array<f_t, 4> tolerances_;
  const std::array<f_t, 4> acceptance_limits_;
  const std::size_t active_count_;
  std::optional<convergence_metrics_t<i_t, f_t>> checkpoint_;
  f_t best_score_ = std::numeric_limits<f_t>::infinity();
  std::array<f_t, 4> envelope_{};
  std::array<std::array<f_t, 4>, progress_window> history_{};
  std::size_t history_next_  = 0;
  std::size_t history_count_ = 0;
};

}  // namespace cuopt::mathematical_optimization::barrier
