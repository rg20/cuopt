/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <barrier/stagnation.hpp>

#include <gtest/gtest.h>

#include <array>
#include <limits>

namespace cuopt::mathematical_optimization::barrier {
namespace {

class BarrierStagnation : public ::testing::Test {
 protected:
  bool observe() { return detector.update(residuals, tolerances, limits, true); }

  stagnation_detector_t<double> detector;
  std::array<double, 4> residuals{1e-10, 1e-10, 1e-10, 2e-4};
  const std::array<double, 4> tolerances{1e-8, 1e-8, 1e-8, 1e-6};
  const std::array<double, 4> limits{1e-4, 1e-4, 1e-4, 1e-3};
};

TEST_F(BarrierStagnation, SustainedPlateau)
{
  for (int i = 0; i < 30; ++i) {
    EXPECT_FALSE(observe());
  }
  EXPECT_TRUE(observe());
}

TEST_F(BarrierStagnation, TemporaryPlateauThenRecovery)
{
  for (int i = 0; i < 29; ++i) {
    EXPECT_FALSE(observe());
  }
  for (int i = 0; i < 150; ++i) {
    residuals[3] *= 0.9;
    EXPECT_FALSE(observe());
  }
}

TEST_F(BarrierStagnation, NonmonotoneProgress)
{
  for (int i = 0; i < 200; ++i) {
    // Large rises between new lows must not cause early acceptance.
    residuals[3] = 2e-4 * std::pow(0.9, i / 20) * ((i % 2) ? 2.0 : 1.0);
    EXPECT_FALSE(observe());
  }
}

TEST_F(BarrierStagnation, SmallImprovementsAccumulate)
{
  for (int i = 0; i < 200; ++i) {
    residuals[3] *= 0.997;
    EXPECT_FALSE(observe());
  }
}

TEST_F(BarrierStagnation, OscillationWithoutProgress)
{
  for (int i = 0; i < 30; ++i) {
    residuals[3] = (i % 2) ? 3e-4 : 2e-4;
    EXPECT_FALSE(observe());
  }
  residuals[3] = 3e-4;
  EXPECT_FALSE(observe());
  residuals[3] = 2e-4;
  EXPECT_TRUE(observe());
}

TEST_F(BarrierStagnation, DeterioratedIterateIsNotAccepted)
{
  EXPECT_FALSE(observe());
  residuals[3] = 4e-4;
  for (int i = 0; i < 100; ++i) {
    EXPECT_FALSE(observe());
  }
  residuals[3] = 2.1e-4;
  EXPECT_TRUE(observe());
}

TEST_F(BarrierStagnation, OtherCriteriaMustStayConverged)
{
  for (int i = 0; i < 25; ++i) {
    EXPECT_FALSE(observe());
  }
  residuals[0] = 2e-8;
  EXPECT_FALSE(observe());
  residuals[0] = 1e-10;
  for (int i = 0; i < 30; ++i) {
    EXPECT_FALSE(observe());
  }
  EXPECT_TRUE(observe());
}

TEST_F(BarrierStagnation, ChangingCandidateRestartsPatience)
{
  for (int i = 0; i < 25; ++i) {
    EXPECT_FALSE(observe());
  }
  residuals[3] = 1e-10;
  residuals[1] = 5e-5;
  for (int i = 0; i < 30; ++i) {
    EXPECT_FALSE(observe());
  }
  EXPECT_TRUE(observe());
}

TEST_F(BarrierStagnation, EveryCriterionCanStagnate)
{
  for (int metric = 0; metric < 4; ++metric) {
    detector = {};
    residuals.fill(1e-10);
    residuals[metric] = 5e-5;
    for (int i = 0; i < 30; ++i) {
      EXPECT_FALSE(observe());
    }
    EXPECT_TRUE(observe());
  }
}

TEST_F(BarrierStagnation, RejectsPoorAccuracyAndInvalidMetrics)
{
  const std::array<double, 4> invalid{
    1e-3, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), -1.0};
  for (double value : invalid) {
    detector     = {};
    residuals[3] = value;
    for (int i = 0; i < 100; ++i) {
      EXPECT_FALSE(observe());
    }
  }
  residuals[3] = 2e-4;
  residuals[0] = std::numeric_limits<double>::quiet_NaN();
  for (int i = 0; i < 100; ++i) {
    EXPECT_FALSE(observe());
  }
}

TEST_F(BarrierStagnation, InactiveObjectiveGapIsIgnored)
{
  residuals[0] = 5e-5;
  residuals[3] = std::numeric_limits<double>::quiet_NaN();
  for (int i = 0; i < 30; ++i) {
    EXPECT_FALSE(detector.update(residuals, tolerances, limits, false));
  }
  EXPECT_TRUE(detector.update(residuals, tolerances, limits, false));
}

TEST_F(BarrierStagnation, FullyConvergedIsNotStagnated)
{
  residuals.fill(0.0);
  for (int i = 0; i < 100; ++i) {
    EXPECT_FALSE(observe());
  }
}

}  // namespace
}  // namespace cuopt::mathematical_optimization::barrier
