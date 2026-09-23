/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <dual_simplex/scaling.hpp>
#include <linear_algebra/sparse_matrix.hpp>

#include <cmath>

namespace cuopt::mathematical_optimization::simplex {

template <typename i_t, typename f_t>
i_t scaling(const lp_problem_t<i_t, f_t>& unscaled,
            const simplex_solver_settings_t<i_t, f_t>& settings,
            lp_problem_t<i_t, f_t>& scaled,
            std::vector<f_t>& column_scaling,
            std::vector<f_t>& row_scaling,
            f_t& objective_rescaling)
{
  scaled = unscaled;
  i_t m  = scaled.num_rows;
  i_t n  = scaled.num_cols;

  row_scaling.assign(m, 1.0);
  objective_rescaling = f_t(1);

  // =========================================================================
  // Ruiz equilibration for SOCP (and QP) problems
  // =========================================================================
  // For SOCP and QP problems, apply Ruiz equilibration to bring the
  // constraint matrix A and Hessian Q close to equilibrium. This improves
  // the conditioning of the KKT system. The skip heuristic considers
  // imbalance in both A and Q.
  if (!unscaled.second_order_cone_dims.empty() || unscaled.Q.n > 0) {
    // col_scale and row_scale accumulate reciprocal scale factors during Ruiz iterations.
    std::vector<f_t> col_scale(n, 1.0);

    // Decide whether Ruiz scaling is needed by checking row- and column-norm
    // imbalance. If both max_norm / min_norm ratios are small, the matrix is
    // already well-conditioned and scaling can hurt (e.g. by amplifying tiny
    // noise coefficients).
    csr_matrix_t<i_t, f_t> Arow_check(0, 0, 0);
    scaled.A.to_compressed_row(Arow_check);
    f_t max_row_norm = 0;
    f_t min_row_norm = std::numeric_limits<f_t>::max();
    for (i_t i = 0; i < m; ++i) {
      f_t row_norm = 0;
      for (i_t p = Arow_check.row_start[i]; p < Arow_check.row_start[i + 1]; ++p) {
        f_t a = std::abs(Arow_check.x[p]);
        if (a > row_norm) row_norm = a;
      }
      if (row_norm > 0) {
        max_row_norm = std::max(max_row_norm, row_norm);
        min_row_norm = std::min(min_row_norm, row_norm);
      }
    }
    f_t row_norm_ratio = (min_row_norm > 0) ? max_row_norm / min_row_norm : 1.0;

    f_t max_col_norm = 0;
    f_t min_col_norm = std::numeric_limits<f_t>::max();
    for (i_t j = 0; j < n; ++j) {
      f_t col_norm = 0;
      for (i_t p = scaled.A.col_start[j]; p < scaled.A.col_start[j + 1]; ++p) {
        f_t a = std::abs(scaled.A.x[p]);
        if (a > col_norm) col_norm = a;
      }
      if (col_norm > 0) {
        max_col_norm = std::max(max_col_norm, col_norm);
        min_col_norm = std::min(min_col_norm, col_norm);
      }
    }
    f_t col_norm_ratio = (min_col_norm > 0) ? max_col_norm / min_col_norm : 1.0;

    // Check coefficient-magnitude spread of Q for the skip heuristic.
    f_t max_q_coeff = 0;
    f_t min_q_coeff = std::numeric_limits<f_t>::max();
    if (scaled.Q.n > 0) {
      for (i_t p = 0; p < scaled.Q.row_start[scaled.Q.m]; ++p) {
        f_t a = std::abs(scaled.Q.x[p]);
        if (a > 0) {
          max_q_coeff = std::max(max_q_coeff, a);
          min_q_coeff = std::min(min_q_coeff, a);
        }
      }
    }
    f_t q_ratio = (min_q_coeff <= max_q_coeff) ? max_q_coeff / min_q_coeff : 1.0;

    // qcqp_ruiz_equilibration: -1 automatic (imbalance heuristic), 0 force off, 1 force on.
    const i_t ruiz_mode  = settings.qcqp_ruiz_equilibration;
    const bool balanced  = row_norm_ratio < 100.0 && col_norm_ratio < 5e4 && q_ratio < 100.0;
    const bool skip_ruiz = (ruiz_mode == 0) || (ruiz_mode < 0 && balanced);

    if (skip_ruiz) {
      if (ruiz_mode == 0) {
        settings.log.printf("Skipping Ruiz equilibration (qcqp_hyper_ruiz_equilibration = 0)\n");
      } else {
        settings.log.printf(
          "Skipping Ruiz equilibration (row norm ratio %.1f, column norm ratio %.1f < 5e4, Q coeff "
          "ratio "
          "%.1f < 100)\n",
          row_norm_ratio,
          col_norm_ratio,
          q_ratio);
      }
      column_scaling.assign(n, 1.0);
      return 0;
    }
    if (ruiz_mode > 0) {
      settings.log.printf(
        "Applying Ruiz equilibration (qcqp_hyper_ruiz_equilibration = 1, row norm ratio %.1f, "
        "column norm ratio %.1f, Q coeff ratio %.1f)\n",
        row_norm_ratio,
        col_norm_ratio,
        q_ratio);
    }

    // -----------------------------------------------------------------------
    // Bound-magnitude column pre-scaling.
    // -----------------------------------------------------------------------
    // Ruiz equilibration only balances the *coefficient* magnitudes of A and Q.
    // On problems whose variable BOUNDS span many orders of magnitude (e.g. a
    // network-flow QP with |bound| ranging from ~1e4 to ~1e11), the constraint
    // matrix can already be perfectly balanced (all +/-1) while the variables
    // themselves live at wildly different scales. The interior-point diagonal
    // D = z/x then spans those same many orders of magnitude, which degrades
    // the conditioning of the KKT factorization and can stall convergence.
    //
    // We fix this by first scaling each column so that the variable it
    // represents becomes O(1): c0[j] = (geometric mean of the finite bound
    // magnitudes of x_j). Substituting x_j = c0[j] * x'_j leaves the feasible
    // region shape unchanged but brings every variable to a common scale, which
    // Ruiz then finishes off on the coefficient side. Columns with no finite,
    // nonzero bound are left at scale 1.
    //
    // Bounds use +/-inf for unbounded sides (see types.hpp). Finite bounds at
    // or beyond finite_bound_limit are treated as unbounded for scaling so we
    // do not overflow when dividing a huge limit by a small scale factor.
    constexpr f_t finite_bound_limit = 1e20;
    {
      std::vector<f_t> c0(n, 1.0);
      std::vector<char> has_bound(n, 0);
      const i_t cone_start0 = unscaled.second_order_cone_dims.empty() ? n : unscaled.cone_var_start;
      f_t geo_sum           = 0.0;
      i_t geo_count         = 0;
      for (i_t j = 0; j < cone_start0; ++j) {
        f_t lo = std::abs(scaled.lower[j]);
        f_t hi = std::abs(scaled.upper[j]);
        f_t mag;
        const bool has_nonzero_finite_lower = scaled.lower[j] > -finite_bound_limit && lo > 0;
        const bool has_nonzero_finite_upper = scaled.upper[j] < finite_bound_limit && hi > 0;
        if (has_nonzero_finite_lower && has_nonzero_finite_upper) {
          mag = std::sqrt(lo * hi);
        } else if (has_nonzero_finite_lower) {
          mag = lo;
        } else if (has_nonzero_finite_upper) {
          mag = hi;
        } else {
          continue;  // free / one-sided-zero: leave at scale 1
        }
        c0[j]        = mag;
        has_bound[j] = 1;
        geo_sum += std::log(mag);
        geo_count++;
      }
      if (geo_count > 0) {
        // Normalize so the average column scale is 1, keeping the overall
        // problem magnitude centered rather than uniformly shrinking it.
        const f_t geo_mean = std::exp(geo_sum / static_cast<f_t>(geo_count));
        for (i_t j = 0; j < cone_start0; ++j) {
          if (has_bound[j]) c0[j] /= geo_mean;
        }
        // Apply x_j = c0[j] * x'_j : A(:,j) *= c0[j], obj[j] *= c0[j],
        // bounds /= c0[j], Q(i,j) *= c0[i]*c0[j], accumulate into col_scale.
        for (i_t j = 0; j < n; ++j) {
          if (c0[j] == 1.0) continue;
          for (i_t p = scaled.A.col_start[j]; p < scaled.A.col_start[j + 1]; ++p) {
            scaled.A.x[p] *= c0[j];
          }
          scaled.objective[j] *= c0[j];
          if (scaled.lower[j] > -finite_bound_limit) scaled.lower[j] /= c0[j];
          if (scaled.upper[j] < finite_bound_limit) scaled.upper[j] /= c0[j];
          col_scale[j] *= c0[j];
        }
        if (scaled.Q.n > 0) {
          for (i_t row = 0; row < scaled.Q.m; ++row) {
            for (i_t p = scaled.Q.row_start[row]; p < scaled.Q.row_start[row + 1]; ++p) {
              i_t col = scaled.Q.j[p];
              scaled.Q.x[p] *= c0[row] * c0[col];
            }
          }
        }
      }
    }

    // Apply Ruiz equilibration
    csr_matrix_t<i_t, f_t> Arow(0, 0, 0);
    scaled.A.to_compressed_row(Arow);

    constexpr i_t max_ruiz_iterations = 10;
    for (i_t iter = 0; iter < max_ruiz_iterations; ++iter) {
      f_t max_deviation = 0.0;

      // --- Row scaling: scale each row by 1/sqrt(max|a_ij|) ---
      std::vector<f_t> r(m);
      for (i_t i = 0; i < m; ++i) {
        f_t rm = 0.0;
        for (i_t p = Arow.row_start[i]; p < Arow.row_start[i + 1]; ++p) {
          f_t a = std::abs(Arow.x[p]);
          if (a > rm) rm = a;
        }
        r[i]          = rm > 0 ? 1.0 / std::sqrt(rm) : 1.0;
        max_deviation = std::max(max_deviation, std::abs(rm - 1.0));
      }
      for (i_t j = 0; j < n; ++j) {
        for (i_t p = scaled.A.col_start[j]; p < scaled.A.col_start[j + 1]; ++p) {
          scaled.A.x[p] *= r[scaled.A.i[p]];
        }
      }
      for (i_t i = 0; i < m; ++i) {
        for (i_t p = Arow.row_start[i]; p < Arow.row_start[i + 1]; ++p) {
          Arow.x[p] *= r[i];
        }
        scaled.rhs[i] *= r[i];
        row_scaling[i] *= r[i];
      }

      // --- Column scaling: scale each column by 1/sqrt(max|a_ij|) ---
      // For cone variables, use a uniform scale per cone block to preserve SOC structure.
      std::vector<f_t> c(n);
      const i_t cone_start = unscaled.second_order_cone_dims.empty() ? n : unscaled.cone_var_start;

      // Linear columns: scale independently. Include the Hessian Q's
      // contribution (row j of Q, symmetric) alongside A's column j, so that
      // large quadratic coefficients are equilibrated too, not just the
      // constraint matrix.
      for (i_t j = 0; j < cone_start; ++j) {
        f_t cm = 0.0;
        for (i_t p = scaled.A.col_start[j]; p < scaled.A.col_start[j + 1]; ++p) {
          f_t a = std::abs(scaled.A.x[p]);
          if (a > cm) cm = a;
        }
        if (scaled.Q.n > 0) {
          for (i_t p = scaled.Q.row_start[j]; p < scaled.Q.row_start[j + 1]; ++p) {
            f_t a = std::abs(scaled.Q.x[p]);
            if (a > cm) cm = a;
          }
        }
        c[j]          = cm > 0 ? 1.0 / std::sqrt(cm) : 1.0;
        max_deviation = std::max(max_deviation, std::abs(cm - 1.0));
      }

      // Cone columns: uniform scale per cone block
      i_t cone_off = cone_start;
      for (i_t k = 0; k < static_cast<i_t>(unscaled.second_order_cone_dims.size()); ++k) {
        i_t q_k = unscaled.second_order_cone_dims[k];
        // Find max column inf-norm across all columns in this cone
        f_t cone_max = 0.0;
        for (i_t j = cone_off; j < cone_off + q_k; ++j) {
          for (i_t p = scaled.A.col_start[j]; p < scaled.A.col_start[j + 1]; ++p) {
            f_t a = std::abs(scaled.A.x[p]);
            if (a > cone_max) cone_max = a;
          }
        }
        f_t cone_scale = cone_max > 0 ? 1.0 / std::sqrt(cone_max) : 1.0;
        max_deviation  = std::max(max_deviation, std::abs(cone_max - 1.0));
        for (i_t j = cone_off; j < cone_off + q_k; ++j) {
          c[j] = cone_scale;
        }
        cone_off += q_k;
      }
      for (i_t j = 0; j < n; ++j) {
        for (i_t p = scaled.A.col_start[j]; p < scaled.A.col_start[j + 1]; ++p) {
          scaled.A.x[p] *= c[j];
        }
      }
      for (i_t i = 0; i < m; ++i) {
        for (i_t p = Arow.row_start[i]; p < Arow.row_start[i + 1]; ++p) {
          Arow.x[p] *= c[Arow.j[p]];
        }
      }
      for (i_t j = 0; j < n; ++j) {
        scaled.objective[j] *= c[j];
        col_scale[j] *= c[j];
      }
      for (i_t j = 0; j < n; ++j) {
        if (scaled.lower[j] > -finite_bound_limit) scaled.lower[j] /= c[j];
        if (scaled.upper[j] < finite_bound_limit) scaled.upper[j] /= c[j];
      }
      if (scaled.Q.n > 0) {
        for (i_t row = 0; row < scaled.Q.m; ++row) {
          for (i_t p = scaled.Q.row_start[row]; p < scaled.Q.row_start[row + 1]; ++p) {
            i_t col = scaled.Q.j[p];
            scaled.Q.x[p] *= c[row] * c[col];
          }
        }
      }
      if (max_deviation < 0.1) break;
    }

    // Ruiz col_scale/row_scaling accumulate reciprocals (c[j] = 1/sqrt(norm)).
    // Invert to match the output convention: C(j,j) = 1/column_scaling[j],
    // R(i,i) = 1/row_scaling[i].
    column_scaling.resize(n);
    for (i_t j = 0; j < n; ++j) {
      column_scaling[j] = f_t(1) / col_scale[j];
    }
    for (i_t i = 0; i < m; ++i) {
      row_scaling[i] = f_t(1) / row_scaling[i];
    }

    f_t a_min = std::numeric_limits<f_t>::max();
    f_t a_max = 0;
    for (i_t p = 0; p < scaled.A.col_start[n]; ++p) {
      f_t a = std::abs(scaled.A.x[p]);
      if (a > 0) {
        a_min = std::min(a_min, a);
        a_max = std::max(a_max, a);
      }
    }
    settings.log.printf("Ruiz equilibration: coefficient range [%e, %e]\n", a_min, a_max);
    return 0;
  }

  if (!settings.scale_columns) {
    settings.log.printf("Skipping column scaling\n");
    column_scaling.resize(n, 1.0);
    return 0;
  }

  if (settings.barrier && settings.barrier_curtis_reid_scaling) {
    // =========================================================================
    // Curtis-Reid + Pock-Chambolle scaling for barrier on plain LPs
    // =========================================================================
    // Ported from PDLP's initial scaling pipeline (pdlp/initial_scaling_strategy/
    // initial_scaling.cu): a log-domain least-squares row/column fit (Curtis-Reid),
    // followed by a single alpha=1 Pock-Chambolle pass. Replaces the plain 2-norm
    // column-only scaling below.
    csr_matrix_t<i_t, f_t> Arow(0, 0, 0);
    scaled.A.to_compressed_row(Arow);

    constexpr f_t finite_bound_limit = 1e20;
    constexpr f_t min_abs_coeff      = std::numeric_limits<f_t>::min();

    // --- Curtis-Reid: alternating row/column log-domain least-squares fit ---
    std::vector<f_t> row_log_scale(m, 0.0);
    std::vector<f_t> col_log_scale(n, 0.0);
    constexpr i_t number_of_curtis_reid_iterations = 10;
    for (i_t iter = 0; iter < number_of_curtis_reid_iterations; ++iter) {
      for (i_t i = 0; i < m; ++i) {
        const i_t rs = Arow.row_start[i];
        const i_t re = Arow.row_start[i + 1];
        if (re == rs) continue;
        f_t sum = 0.0;
        for (i_t p = rs; p < re; ++p) {
          f_t abs_val = std::max(std::abs(Arow.x[p]), min_abs_coeff);
          sum += -std::log(abs_val) - col_log_scale[Arow.j[p]];
        }
        row_log_scale[i] = sum / static_cast<f_t>(re - rs);
      }
      for (i_t j = 0; j < n; ++j) {
        const i_t cs = scaled.A.col_start[j];
        const i_t ce = scaled.A.col_start[j + 1];
        if (ce == cs) continue;
        f_t sum = 0.0;
        for (i_t p = cs; p < ce; ++p) {
          f_t abs_val = std::max(std::abs(scaled.A.x[p]), min_abs_coeff);
          sum += -std::log(abs_val) - row_log_scale[scaled.A.i[p]];
        }
        col_log_scale[j] = sum / static_cast<f_t>(ce - cs);
      }
    }

    // Fold the log-domain fit into multiplicative row/column scale factors.
    // clamp_bound = 30 (exp(+-30) ~ [9.4e-14, 1.07e13]) bounds the range a
    // pathological fit could produce, matching PDLP's Curtis-Reid.
    constexpr f_t clamp_bound = f_t(30);
    auto exp_clamped          = [clamp_bound](f_t v) {
      return std::exp(std::min(std::max(v, -clamp_bound), clamp_bound));
    };
    std::vector<f_t> row_scale(m), col_scale(n);
    for (i_t i = 0; i < m; ++i)
      row_scale[i] = exp_clamped(row_log_scale[i]);
    for (i_t j = 0; j < n; ++j)
      col_scale[j] = exp_clamped(col_log_scale[j]);

    for (i_t j = 0; j < n; ++j) {
      for (i_t p = scaled.A.col_start[j]; p < scaled.A.col_start[j + 1]; ++p) {
        scaled.A.x[p] *= row_scale[scaled.A.i[p]] * col_scale[j];
      }
    }
    for (i_t i = 0; i < m; ++i) {
      for (i_t p = Arow.row_start[i]; p < Arow.row_start[i + 1]; ++p) {
        Arow.x[p] *= row_scale[i] * col_scale[Arow.j[p]];
      }
      scaled.rhs[i] *= row_scale[i];
    }
    for (i_t j = 0; j < n; ++j) {
      scaled.objective[j] *= col_scale[j];
      if (scaled.lower[j] > -finite_bound_limit) scaled.lower[j] /= col_scale[j];
      if (scaled.upper[j] < finite_bound_limit) scaled.upper[j] /= col_scale[j];
    }

    // --- Pock-Chambolle: single alpha=1 pass balancing row/column norms ---
    constexpr f_t alpha = f_t(1);
    std::vector<f_t> row_weight(m, 0.0), col_weight(n, 0.0);
    for (i_t i = 0; i < m; ++i) {
      f_t sum = 0.0;
      for (i_t p = Arow.row_start[i]; p < Arow.row_start[i + 1]; ++p) {
        sum += std::pow(std::abs(Arow.x[p]), alpha);
      }
      row_weight[i] = sum;
    }
    for (i_t j = 0; j < n; ++j) {
      f_t sum = 0.0;
      for (i_t p = scaled.A.col_start[j]; p < scaled.A.col_start[j + 1]; ++p) {
        sum += std::pow(std::abs(scaled.A.x[p]), f_t(2) - alpha);
      }
      col_weight[j] = sum;
    }
    std::vector<f_t> row_pc(m, 1.0), col_pc(n, 1.0);
    for (i_t i = 0; i < m; ++i) {
      if (row_weight[i] > 0) row_pc[i] = f_t(1) / std::sqrt(row_weight[i]);
    }
    for (i_t j = 0; j < n; ++j) {
      if (col_weight[j] > 0) col_pc[j] = f_t(1) / std::sqrt(col_weight[j]);
    }

    for (i_t j = 0; j < n; ++j) {
      for (i_t p = scaled.A.col_start[j]; p < scaled.A.col_start[j + 1]; ++p) {
        scaled.A.x[p] *= row_pc[scaled.A.i[p]] * col_pc[j];
      }
    }
    for (i_t i = 0; i < m; ++i) {
      scaled.rhs[i] *= row_pc[i];
      row_scale[i] *= row_pc[i];
    }
    for (i_t j = 0; j < n; ++j) {
      scaled.objective[j] *= col_pc[j];
      if (scaled.lower[j] > -finite_bound_limit) scaled.lower[j] /= col_pc[j];
      if (scaled.upper[j] < finite_bound_limit) scaled.upper[j] /= col_pc[j];
      col_scale[j] *= col_pc[j];
    }

    // Match the output convention used by the other branches: column_scaling[j] /
    // row_scaling[i] are the factors to divide the *scaled* problem by to recover
    // the unscaled one, i.e. 1 / (applied multiplicative factor).
    column_scaling.resize(n);
    for (i_t j = 0; j < n; ++j)
      column_scaling[j] = f_t(1) / col_scale[j];
    for (i_t i = 0; i < m; ++i)
      row_scaling[i] = f_t(1) / row_scale[i];

    // --- Bound + objective rescaling (ported from PDLP's bound_objective_rescaling) ---
    // A whole-problem scalar balance, layered on top of the row/column scaling above:
    // bound_rescale (k) uniformly rescales the RHS and variable bounds (equivalent to a
    // uniform column scale, so it folds directly into column_scaling); objective_rescale
    // (m) rescales the objective only. Since it touches the dual (y) and reduced cost (z)
    // symmetrically -- independent of column_scaling's own, unrelated effect on x and z --
    // it cannot be folded into column_scaling or row_scaling and is returned separately;
    // see unscale_solution.
    f_t bound_sum_sq = 0.0;
    for (i_t i = 0; i < m; ++i)
      bound_sum_sq += scaled.rhs[i] * scaled.rhs[i];
    const f_t bound_rescale = f_t(1) / (std::sqrt(bound_sum_sq) + f_t(1));

    f_t obj_sum_sq = 0.0;
    for (i_t j = 0; j < n; ++j)
      obj_sum_sq += scaled.objective[j] * scaled.objective[j];
    const f_t objective_rescale = f_t(1) / (std::sqrt(obj_sum_sq) + f_t(1));

    for (i_t i = 0; i < m; ++i) {
      scaled.rhs[i] *= bound_rescale;
    }
    for (i_t j = 0; j < n; ++j) {
      if (scaled.lower[j] > -finite_bound_limit) scaled.lower[j] *= bound_rescale;
      if (scaled.upper[j] < finite_bound_limit) scaled.upper[j] *= bound_rescale;
      scaled.objective[j] *= objective_rescale;
      column_scaling[j] *= bound_rescale;
    }
    objective_rescaling = objective_rescale;

    // compute_user_objective()/compute_presolved_objective() (dual_simplex/solve.cpp) convert
    // between the LP's own objective terms (c^Tx computed in *this* lp's coordinate system) and
    // the true user objective via obj_scale/obj_constant. Since bound_rescale/objective_rescale
    // multiply c^Tx by (bound_rescale * objective_rescale) relative to unscaled, fold the inverse
    // into scaled.obj_scale/obj_constant here so every caller downstream (barrier.cu's internal
    // convergence checks and final reporting alike) sees the correct true-scale objective without
    // needing to know about this rescale.
    const f_t combined_rescale = bound_rescale * objective_rescale;
    scaled.obj_scale /= combined_rescale;
    scaled.obj_constant *= combined_rescale;

    settings.log.printf(
      "Bound rescaling %e, objective rescaling %e\n", bound_rescale, objective_rescale);

    f_t a_min = std::numeric_limits<f_t>::max();
    f_t a_max = 0;
    for (i_t p = 0; p < scaled.A.col_start[n]; ++p) {
      f_t a = std::abs(scaled.A.x[p]);
      if (a > 0) {
        a_min = std::min(a_min, a);
        a_max = std::max(a_max, a);
      }
    }
    settings.log.printf(
      "Curtis-Reid + Pock-Chambolle scaling: coefficient range [%e, %e]\n", a_min, a_max);
    return 0;
  }

  column_scaling.resize(n);
  f_t max = 0;
  f_t min = std::numeric_limits<f_t>::max();
  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = scaled.A.col_start[j];
    const i_t col_end   = scaled.A.col_start[j + 1];
    f_t sum             = 0.0;
    for (i_t p = col_start; p < col_end; ++p) {
      const f_t x = scaled.A.x[p];
      sum += x * x;
    }
    f_t col_norm_j = column_scaling[j] = sum > 0 ? std::sqrt(sum) : 1.0;
    max                                = std::max(col_norm_j, max);
    min                                = std::min(col_norm_j, min);
  }
  settings.log.printf("Scaling matrix. Maximum column norm %e, minimum column norm %e\n", max, min);
  // C(j, j) = 1/column_scaling(j)

  // scaled_A = unscaled_A * C
  for (i_t j = 0; j < n; ++j) {
    const i_t col_start = scaled.A.col_start[j];
    const i_t col_end   = scaled.A.col_start[j + 1];
    for (i_t p = col_start; p < col_end; ++p) {
      scaled.A.x[p] /= column_scaling[j];
    }
  }
  // scaled_obj = C*unscaled_obj
  for (i_t j = 0; j < n; ++j) {
    scaled.objective[j] /= column_scaling[j];
  }
  // scaled_lower = C^{-1} * unscaled_lower
  // scaled_upper = C^{-1} * unscaled_upper
  for (i_t j = 0; j < n; ++j) {
    scaled.lower[j] *= column_scaling[j];
    scaled.upper[j] *= column_scaling[j];
  }

  for (i_t i = 0; i < unscaled.Q.n; ++i) {
    const i_t row_start = unscaled.Q.row_start[i];
    const i_t row_end   = unscaled.Q.row_start[i + 1];
    i_t row             = i;
    for (i_t p = row_start; p < row_end; ++p) {
      i_t col       = unscaled.Q.j[p];
      scaled.Q.x[p] = unscaled.Q.x[p] / (column_scaling[row] * column_scaling[col]);
    }
  }
  return 0;
}

template <typename i_t, typename f_t>
void unscale_solution(const std::vector<f_t>& column_scaling,
                      const std::vector<f_t>& row_scaling,
                      f_t objective_rescaling,
                      const std::vector<f_t>& scaled_x,
                      const std::vector<f_t>& scaled_y,
                      const std::vector<f_t>& scaled_z,
                      std::vector<f_t>& unscaled_x,
                      std::vector<f_t>& unscaled_y,
                      std::vector<f_t>& unscaled_z)
{
  const i_t n = scaled_x.size();
  unscaled_x.resize(n);
  unscaled_z.resize(n);
  // z picks up 1/objective_rescaling on top of the usual column_scaling factor: scaling the
  // objective by objective_rescaling scales the reduced cost by the same factor (see the
  // derivation in scaling.hpp), independent of column_scaling's own effect on x and z.
  for (i_t j = 0; j < n; ++j) {
    unscaled_x[j] = scaled_x[j] / column_scaling[j];
    unscaled_z[j] = scaled_z[j] * column_scaling[j] / objective_rescaling;
  }

  const i_t m = scaled_y.size();
  unscaled_y.resize(m);
  // R(i,i) = 1/row_scaling[i], so y_orig = y_scaled / row_scaling; y additionally picks up
  // 1/objective_rescaling for the same reason as z above.
  for (i_t i = 0; i < m; ++i) {
    unscaled_y[i] = scaled_y[i] / (row_scaling[i] * objective_rescaling);
  }
}

#ifdef DUAL_SIMPLEX_INSTANTIATE_DOUBLE

template int scaling<int, double>(const lp_problem_t<int, double>& unscaled,
                                  const simplex_solver_settings_t<int, double>& settings,
                                  lp_problem_t<int, double>& scaled,
                                  std::vector<double>& column_scaling,
                                  std::vector<double>& row_scaling,
                                  double& objective_rescaling);

template void unscale_solution<int, double>(const std::vector<double>& column_scaling,
                                            const std::vector<double>& row_scaling,
                                            double objective_rescaling,
                                            const std::vector<double>& scaled_x,
                                            const std::vector<double>& scaled_y,
                                            const std::vector<double>& scaled_z,
                                            std::vector<double>& unscaled_x,
                                            std::vector<double>& unscaled_y,
                                            std::vector<double>& unscaled_z);

#endif

}  // namespace cuopt::mathematical_optimization::simplex
