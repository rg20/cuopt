/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <barrier/convergence.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace cuopt::mathematical_optimization::barrier {
namespace {

class BarrierConvergence : public ::testing::Test {
 protected:
  convergence_observation_t observe()
  {
    const auto result = monitor.observe(metrics);
    ++metrics.iteration;
    return result;
  }

  const std::array<double, 4> tolerances{1e-8, 1e-8, 1e-8, 1e-6};
  const std::array<double, 4> limits{1e-4, 1e-4, 1e-4, 1e-3};
  convergence_monitor_t<int32_t, double> monitor{tolerances, limits, true};
  convergence_metrics_t<int32_t, double> metrics{
    0, 1.0, 0.99, {1e-5, 2e-5, 1e-8, 1e-2}, {1e-7, 2e-7, 1e-10, 1e-8}};
};

TEST_F(BarrierConvergence, MultipleUnmetCriteriaPlateau)
{
  for (int i = 0; i < 30; ++i) {
    EXPECT_FALSE(observe().stagnated);
  }
  EXPECT_TRUE(observe().stagnated);
  ASSERT_TRUE(monitor.checkpoint());
  EXPECT_EQ(monitor.checkpoint()->iteration, 0);
}

TEST_F(BarrierConvergence, ProgressInNonWorstCriterionRenewsPatience)
{
  metrics.relative = {1e-6, 5e-7, 1e-10, 1e-8};
  for (int i = 0; i < 100; ++i) {
    metrics.relative[1] *= 0.98;
    EXPECT_FALSE(observe().stagnated);
  }
}

TEST_F(BarrierConvergence, BelowToleranceProgressDoesNotRenewPatience)
{
  metrics.relative = {1e-7, 5e-9, 1e-10, 1e-8};
  for (int i = 0; i < 30; ++i) {
    metrics.relative[1] *= 0.5;
    EXPECT_FALSE(observe().stagnated);
  }
  EXPECT_TRUE(observe().stagnated);
}

TEST_F(BarrierConvergence, SmallImprovementsAccumulate)
{
  for (int i = 0; i < 150; ++i) {
    metrics.relative[0] *= 0.997;
    EXPECT_FALSE(observe().stagnated);
  }
}

TEST_F(BarrierConvergence, NonmonotoneProgress)
{
  for (int i = 0; i < 100; ++i) {
    metrics.relative[0] = 1e-6 * std::pow(0.98, i) * (i % 2 ? 2 : 1);
    EXPECT_FALSE(observe().stagnated);
  }
}

TEST_F(BarrierConvergence, OscillationWithoutProgress)
{
  for (int i = 0; i < 30; ++i) {
    metrics.relative[0] = i % 2 ? 2e-7 : 1e-7;
    EXPECT_FALSE(observe().stagnated);
  }
  metrics.relative[0] = 2e-7;
  EXPECT_TRUE(observe().stagnated);
}

TEST_F(BarrierConvergence, CheckpointIsCompleteRealIterate)
{
  metrics.relative = {1e-7, 1e-6, 1e-10, 1e-8};
  EXPECT_TRUE(observe().save_checkpoint);
  metrics.relative         = {9e-7, 2e-7, 1e-10, 1e-8};
  metrics.absolute         = {0.2, 0.3, 0.4, 0.5};
  metrics.primal_objective = 7;
  metrics.dual_objective   = 6.5;
  EXPECT_TRUE(observe().save_checkpoint);
  ASSERT_TRUE(monitor.checkpoint());
  EXPECT_EQ(monitor.checkpoint()->iteration, 1);
  EXPECT_EQ(monitor.checkpoint()->relative, metrics.relative);
  EXPECT_EQ(monitor.checkpoint()->absolute, metrics.absolute);
  EXPECT_EQ(monitor.checkpoint()->primal_objective, 7);
  EXPECT_EQ(monitor.checkpoint()->dual_objective, 6.5);
}

TEST_F(BarrierConvergence, ObjectiveGapAffectsCheckpointRanking)
{
  metrics.relative = {1e-7, 1e-7, 1e-10, 1e-4};
  EXPECT_TRUE(observe().save_checkpoint);
  metrics.relative = {2e-7, 2e-7, 1e-10, 2e-5};
  EXPECT_TRUE(observe().save_checkpoint);
  EXPECT_NEAR(monitor.best_score(), 20, 1e-12);
  metrics.relative[3] = 2e-4;
  EXPECT_FALSE(observe().save_checkpoint);
}

TEST_F(BarrierConvergence, NoAcceptableCheckpointNoStop)
{
  metrics.relative[3] = limits[3];
  for (int i = 0; i < 100; ++i) {
    EXPECT_FALSE(observe().stagnated);
  }
  EXPECT_FALSE(monitor.checkpoint());
}

TEST_F(BarrierConvergence, DeteriorationPreservesAcceptableCheckpoint)
{
  EXPECT_TRUE(observe().save_checkpoint);
  metrics.relative[0] = 1;
  for (int i = 0; i < 29; ++i) {
    EXPECT_FALSE(observe().stagnated);
  }
  EXPECT_TRUE(observe().stagnated);
  EXPECT_EQ(monitor.checkpoint()->iteration, 0);
}

TEST_F(BarrierConvergence, StrictConvergenceOverridesPatience)
{
  for (int i = 0; i < 30; ++i) {
    observe();
  }
  metrics.relative  = {1e-9, 1e-9, 1e-9, 1e-7};
  const auto result = observe();
  EXPECT_TRUE(result.converged);
  EXPECT_FALSE(result.stagnated);
}

TEST_F(BarrierConvergence, ToleranceCrossingRenewsPatience)
{
  metrics.relative[0] = 1.01e-8;
  for (int i = 0; i < 30; ++i) {
    observe();
  }
  metrics.relative[0] = 0.999e-8;
  EXPECT_FALSE(observe().stagnated);
}

TEST_F(BarrierConvergence, InvalidMetricsDoNotChangeState)
{
  EXPECT_TRUE(observe().save_checkpoint);
  const auto valid = metrics;
  for (const double invalid :
       {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -1.0}) {
    for (int i = 0; i < 4; ++i) {
      metrics             = valid;
      metrics.relative[i] = invalid;
      EXPECT_FALSE(observe().valid);
      metrics             = valid;
      metrics.absolute[i] = invalid;
      EXPECT_FALSE(observe().valid);
    }
  }
  metrics                  = valid;
  metrics.primal_objective = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(observe().valid);
  metrics                = valid;
  metrics.dual_objective = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(observe().valid);
  EXPECT_EQ(monitor.checkpoint()->iteration, 0);
  metrics = valid;
  for (int i = 0; i < 29; ++i) {
    EXPECT_FALSE(observe().stagnated);
  }
  EXPECT_TRUE(observe().stagnated);
}

TEST_F(BarrierConvergence, InactiveGapIsIgnored)
{
  convergence_monitor_t<int32_t, double> lp_monitor(tolerances, limits, false);
  metrics.relative[3] = metrics.absolute[3] = std::numeric_limits<double>::quiet_NaN();
  for (int i = 0; i < 30; ++i) {
    const auto result = lp_monitor.observe(metrics);
    EXPECT_TRUE(result.valid);
    EXPECT_FALSE(result.stagnated);
    ++metrics.iteration;
  }
  EXPECT_TRUE(lp_monitor.observe(metrics).stagnated);
}

TEST_F(BarrierConvergence, InvalidToleranceDisablesObservation)
{
  auto invalid = tolerances;
  invalid[0]   = 0;
  convergence_monitor_t<int32_t, double> invalid_monitor(invalid, limits, true);
  EXPECT_FALSE(invalid_monitor.observe(metrics).valid);
  EXPECT_FALSE(invalid_monitor.checkpoint());
}

TEST_F(BarrierConvergence, ProgressCanResumeAfterPatienceExpires)
{
  for (int i = 0; i < 31; ++i) {
    observe();
  }
  metrics.relative[0] *= 0.9;
  EXPECT_FALSE(observe().stagnated);
}

TEST_F(BarrierConvergence, RecoveryAboveHistoricalBestIsBounded)
{
  metrics.relative = {1e-7, 1e-7, 1e-10, 1e-8};
  EXPECT_TRUE(observe().save_checkpoint);
  // Sustained local recovery cannot extend a full window without making
  // meaningful progress against the historical best.
  for (int i = 0; i < 29; ++i) {
    metrics.relative[0] = 1e-5 * std::pow(0.997, i);
    EXPECT_FALSE(observe().stagnated);
  }
  EXPECT_TRUE(observe().stagnated);
  EXPECT_EQ(monitor.checkpoint()->iteration, 0);
}

TEST_F(BarrierConvergence, ApparentRecoveryCannotHideDivergence)
{
  metrics.relative = {1e-7, 1e-7, 1e-10, 1e-8};
  observe();
  for (int i = 0; i < 29; ++i) {
    metrics.relative[0] = 1e-5 * std::pow(0.98, i);
    metrics.relative[1] = 1e-5 * std::pow(1.02, i);
    EXPECT_FALSE(observe().stagnated);
  }
  EXPECT_TRUE(observe().stagnated);
  EXPECT_EQ(monitor.checkpoint()->iteration, 0);
}

TEST_F(BarrierConvergence, NonmonotoneRecoveryWithinWindow)
{
  metrics.relative = {1e-7, 5e-8, 1e-10, 1e-8};
  observe();
  for (int i = 0; i < 20; ++i) {
    metrics.relative[0] = 2e-7 * std::pow(0.95, i) * (i % 2 ? 2 : 1);
    EXPECT_FALSE(observe().stagnated);
  }
  metrics.relative[0] = 8e-8;
  for (int i = 0; i < 9; ++i) {
    EXPECT_FALSE(observe().stagnated);
  }
  EXPECT_FALSE(observe().stagnated);
  EXPECT_LT(monitor.checkpoint()->relative[0], 1e-7);
}

TEST_F(BarrierConvergence, RecoveryEventuallyPlateaus)
{
  observe();
  for (int i = 0; i < 60; ++i) {
    metrics.relative[0] = 1e-7 * std::pow(0.98, i);
    EXPECT_FALSE(observe().stagnated);
  }
  bool stopped = false;
  for (int i = 0; i < 45; ++i) {
    stopped = stopped || observe().stagnated;
  }
  EXPECT_TRUE(stopped);
}

TEST_F(BarrierConvergence, AllFourCriteriaCanRemainUnmet)
{
  metrics.relative = {1e-7, 2e-7, 3e-7, 4e-5};
  for (int i = 0; i < 30; ++i) {
    EXPECT_FALSE(observe().stagnated);
  }
  EXPECT_TRUE(observe().stagnated);
}

TEST_F(BarrierConvergence, NearConvergedReboundsDoNotRenewWindow)
{
  metrics.relative = {1e-12, 2e-9, 2e-9, 1.09e-6};
  EXPECT_TRUE(observe().save_checkpoint);
  for (int i = 1; i < 30; ++i) {
    metrics.relative[3] = 2e-5 * std::pow(0.99, i);
    EXPECT_FALSE(observe().stagnated);
  }
  EXPECT_TRUE(observe().stagnated);
  EXPECT_EQ(monitor.checkpoint()->iteration, 0);
  EXPECT_DOUBLE_EQ(monitor.checkpoint()->relative[3], 1.09e-6);
}

TEST_F(BarrierConvergence, WindowStartsAtFirstAcceptableCheckpoint)
{
  metrics.relative[3] = 1e-2;
  for (int i = 0; i < 50; ++i) {
    EXPECT_FALSE(observe().stagnated);
  }
  metrics.relative[3] = 1e-4;
  for (int i = 0; i < 30; ++i) {
    EXPECT_FALSE(observe().stagnated);
  }
  EXPECT_TRUE(observe().stagnated);
  EXPECT_EQ(monitor.checkpoint()->iteration, 50);
}

TEST_F(BarrierConvergence, OldProgressLeavesRollingWindow)
{
  observe();
  for (int i = 1; i < 20; ++i) {
    EXPECT_FALSE(observe().stagnated);
  }
  metrics.relative[0] *= 0.9;
  for (int i = 20; i < 50; ++i) {
    EXPECT_FALSE(observe().stagnated);
  }
  EXPECT_TRUE(observe().stagnated);
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::barrier
