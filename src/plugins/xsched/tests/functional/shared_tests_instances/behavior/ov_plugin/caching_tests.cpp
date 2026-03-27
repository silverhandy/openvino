// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/ov_plugin/caching_tests.hpp"

using namespace ov::test::behavior;

namespace {
static const std::vector<ov::element::Type> precisionsXsched = {
    ov::element::f32,
};

static const std::vector<std::size_t> batchSizesXsched = {1, 2};

INSTANTIATE_TEST_SUITE_P(smoke_Behavior_CachingSupportCase_Xsched,
                         CompileModelCacheTestBase,
                         ::testing::Combine(::testing::ValuesIn(CompileModelCacheTestBase::getStandardFunctions()),
                                            ::testing::ValuesIn(precisionsXsched),
                                            ::testing::ValuesIn(batchSizesXsched),
                                            ::testing::Values(ov::test::utils::DEVICE_XSCHED),
                                            ::testing::Values(ov::AnyMap{})),
                         CompileModelCacheTestBase::getTestCaseName);

const std::vector<ov::AnyMap> XschedConfigs = {
    {ov::num_streams(2)},
};
const std::vector<std::string> TestXschedTargets = {
    ov::test::utils::DEVICE_XSCHED,
};
INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Xsched,
                         CompileModelLoadFromMemoryTestBase,
                         ::testing::Combine(::testing::ValuesIn(TestXschedTargets),
                                            ::testing::ValuesIn(XschedConfigs)),
                         CompileModelLoadFromMemoryTestBase::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Xsched,
                         CompileModelLoadFromCacheTest,
                         ::testing::Combine(::testing::ValuesIn(TestXschedTargets),
                                            ::testing::ValuesIn(XschedConfigs)),
                         CompileModelLoadFromCacheTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CachingSupportCase_Xsched,
                         CompileModelWithCacheEncryptionTest,
                         testing::ValuesIn(TestXschedTargets),
                         CompileModelWithCacheEncryptionTest::getTestCaseName);
}  // namespace
