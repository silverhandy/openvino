# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# [cmake:plugin]
set(TARGET_NAME "openvino_xsched_plugin")

if(NOT BUILD_SHARED_LIBS)
    message(STATUS "XSCHED plugin is disabled for static OpenVINO builds (BUILD_SHARED_LIBS=OFF)")
    return()
endif()

if(DEFINED ENABLE_PLUGINS_XML AND NOT ENABLE_PLUGINS_XML)
    message(STATUS "XSCHED plugin is disabled when ENABLE_PLUGINS_XML=OFF")
    return()
endif()

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

if (NOT ENABLE_TEMPLATE_REGISTRATION)
    # Skip install and registration of template component
    set(skip_plugin SKIP_INSTALL SKIP_REGISTRATION)
endif()

# adds a shared library with plugin
ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "XSCHED"
              SOURCES ${SOURCES} ${HEADERS}
              ${skip_plugin}
              VERSION_DEFINES_FOR plugin.cpp
              ADD_CLANG_FORMAT)

# Enable support of CC for the plugin
ov_mark_target_as_cc(${TARGET_NAME})

target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_LIST_DIR}/../include")

set(_xsched_components preempt sched utils shimlevelzero protocol)
set(_xsched_found FALSE)

foreach(_xsched_target IN LISTS _xsched_components)
    if(TARGET XSched::${_xsched_target})
        set(_xsched_found TRUE)
    else()
        set(_xsched_found FALSE)
        break()
    endif()
endforeach()

if(NOT _xsched_found)
    set(_xsched_cmake_dir "${CMAKE_CURRENT_LIST_DIR}/../cmake")
    if(EXISTS "${_xsched_cmake_dir}")
        list(APPEND CMAKE_MODULE_PATH "${_xsched_cmake_dir}")
    endif()

    find_package(XSched QUIET COMPONENTS ${_xsched_components})

    if(NOT XSched_FOUND)
        set(_xsched_source_dir "${CMAKE_CURRENT_LIST_DIR}/../../../../../xsched")
        if(EXISTS "${_xsched_source_dir}/CMakeLists.txt")
            message(STATUS "XSched package not found; building from ${_xsched_source_dir}")
            set(PLATFORM_LEVELZERO ON CACHE BOOL "Build Level Zero platform" FORCE)
            set(SHIM_SOFTLINK OFF CACHE BOOL "Disable shim softlinks" FORCE)
            set(BUILD_TEST OFF CACHE BOOL "Disable XSched tests" FORCE)
            add_subdirectory("${_xsched_source_dir}" "${CMAKE_BINARY_DIR}/xsched" EXCLUDE_FROM_ALL)
            set(XSched_FOUND TRUE)
        endif()
    endif()

    foreach(_xsched_target IN LISTS _xsched_components)
        if(TARGET XSched::${_xsched_target})
            set(_xsched_found TRUE)
        else()
            set(_xsched_found FALSE)
            break()
        endif()
    endforeach()
endif()

if(NOT _xsched_found)
    message(FATAL_ERROR
        "XSched components not found. Set XSched_DIR to an existing installation "
        "or build 3rd-party/xsched before configuring the XSCHED OpenVINO plugin.")
endif()

# link common OpenVINO Runtime libraries
target_link_libraries(${TARGET_NAME} PRIVATE
    openvino::interpreter_backend
    openvino::reference
    XSched::preempt
    XSched::sched
    XSched::utils
    XSched::shimlevelzero
    XSched::protocol)

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if (ENABLE_TEMPLATE_REGISTRATION)
    # Update the plugins.xml file
    ov_register_plugins(MAIN_TARGET ${TARGET_NAME})
endif()
# [cmake:plugin]

install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
