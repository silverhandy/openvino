# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:main]
cmake_minimum_required(VERSION 3.13)

project(OpenVINOXschedPlugin)

if(NOT OpenVINODeveloperPackage_DIR)
    get_filename_component(_ov_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
    file(GLOB _ov_devpkg_config
         "${_ov_repo_root}/build/OpenVINODeveloperPackageConfig.cmake"
         "${_ov_repo_root}/build/share/OpenVINODeveloperPackageConfig.cmake"
         "${_ov_repo_root}/build/build-modules/OpenVINODeveloperPackageConfig.cmake")
    list(LENGTH _ov_devpkg_config _ov_devpkg_config_count)
    if(_ov_devpkg_config_count GREATER 0)
        list(GET _ov_devpkg_config 0 _ov_devpkg_config_file)
        get_filename_component(_ov_devpkg_config_dir "${_ov_devpkg_config_file}" DIRECTORY)
        set(OpenVINODeveloperPackage_DIR "${_ov_devpkg_config_dir}" CACHE PATH
            "Path to the OpenVINO developer package" FORCE)
        message(STATUS "OpenVINO developer package detected at ${OpenVINODeveloperPackage_DIR}")
    endif()
    unset(_ov_devpkg_config)
    unset(_ov_devpkg_config_count)
    unset(_ov_devpkg_config_file)
    unset(_ov_devpkg_config_dir)
    unset(_ov_repo_root)
endif()

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

ov_option(ENABLE_XSCHED_REGISTRATION "Enables registration of XSCHED plugin" OFF)

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
    ov_cpack(xsched)
endif()
# [cmake:main]
