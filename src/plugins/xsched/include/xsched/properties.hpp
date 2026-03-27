// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced properties for the XSched plugin.
 * These properties should be used in set_property() and compile_model() methods of plugins
 *
 * @file xsched/properties.hpp
 */

#pragma once

#include <string>

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace xsched_plugin {

// ! [properties:public_header]

/**
 * @brief Allows disabling all transformations for execution inside the XSCHED plugin.
 */
static constexpr Property<bool, PropertyMutability::RW> disable_transformations{"DISABLE_TRANSFORMATIONS"};

// ! [properties:public_header]

}  // namespace xsched_plugin
}  // namespace ov
