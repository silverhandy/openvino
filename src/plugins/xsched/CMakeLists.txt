# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:main]
cmake_minimum_required(VERSION 3.13)

project(OpenVINOXschedPlugin)

if(NOT COMMAND ov_add_plugin)
    find_package(OpenVINODeveloperPackage REQUIRED)
endif()

if(NOT BUILD_SHARED_LIBS)
    message(STATUS "XSCHED plugin is skipped (requires BUILD_SHARED_LIBS=ON)")
    return()
endif()

if(DEFINED ENABLE_PLUGINS_XML AND NOT ENABLE_PLUGINS_XML)
    message(STATUS "XSCHED plugin is skipped (requires ENABLE_PLUGINS_XML=ON)")
    return()
endif()

ov_option(ENABLE_TEMPLATE_REGISTRATION "Enables registration of TEMPLATE plugin" OFF)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wall)
endif()

add_subdirectory(src)

if(TARGET openvino_xsched_plugin AND ENABLE_TESTS)
    include(CTest)
    enable_testing()

    if(ENABLE_FUNCTIONAL_TESTS)
        add_subdirectory(tests/functional)
    endif()
endif()

# install

if(OpenVINODeveloperPackage_FOUND)
    ov_cpack(template)
endif()
# [cmake:main]
