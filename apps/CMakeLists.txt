# SPDX-FileCopyrightText: 2025 Rahil Miten Doshi, Friedrich-Alexander-Universität Erlangen-Nürnberg
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Top-level driver for the MaterForge demonstration apps. waLBerla is added once
# here (shared by every app) and each app lives in its own subdirectory, so a
# single configure builds both. These apps depend on waLBerla and pystencils
# (GPLv3), which is why everything under apps/ is GPL-licensed (see ./LICENSE).

cmake_minimum_required(VERSION 3.24)
project(materforge_apps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Which apps to configure. Each app is independently switchable, so you can build
# just one - e.g. -DBUILD_HEAT_EQUATION=OFF for a CouetteFlow-only GPU build.
option(BUILD_HEAT_EQUATION "Build the HeatEquationKernel demo app" ON)
option(BUILD_COUETTE_FLOW  "Build the CouetteFlow benchmark app"   ON)

# ============================================================================
# TARGET PLATFORM (shared: drives the single waLBerla build)
#   CPU (default), GPU-HIP, or GPU-CUDA. GPU is opt-in via a preset or -D flag
#   so a plain `cmake` on a CPU box - the HeatEquationKernel demo's home - just
#   works without a GPU toolchain present.
# ============================================================================
set(TARGET_PLATFORM "CPU" CACHE STRING "Target platform: CPU, GPU-HIP, or GPU-CUDA")
set_property(CACHE TARGET_PLATFORM PROPERTY STRINGS "CPU" "GPU-HIP" "GPU-CUDA")
message(STATUS "TARGET_PLATFORM: ${TARGET_PLATFORM}")

# Languages must be enabled before find_package / add_subdirectory.
if(TARGET_PLATFORM STREQUAL "GPU-HIP")
    enable_language(HIP)
    set(CMAKE_HIP_ARCHITECTURES "gfx90a")
    set(AMDGPU_TARGETS "gfx90a")
    message(STATUS "HIP language enabled for gfx90a")
elseif(TARGET_PLATFORM STREQUAL "GPU-CUDA")
    enable_language(CUDA)
    message(STATUS "CUDA language enabled")
endif()

# ============================================================================
# WALBERLA OPTIONS (must precede add_subdirectory(walberla))
# ============================================================================
set(WALBERLA_BUILD_TESTS      OFF CACHE BOOL "")
set(WALBERLA_BUILD_BENCHMARKS OFF CACHE BOOL "")
set(WALBERLA_BUILD_TOOLS      OFF CACHE BOOL "")
set(WALBERLA_BUILD_TUTORIALS  OFF CACHE BOOL "")
set(WALBERLA_BUILD_SHOWCASES  OFF CACHE BOOL "")
set(WALBERLA_BUILD_DOC        OFF CACHE BOOL "")
set(WALBERLA_BUILD_EXAMPLES   OFF CACHE BOOL "")
# Only SweepGen (code generation) is needed - not the Python bindings.
set(WALBERLA_BUILD_WITH_PYTHON OFF CACHE BOOL "" FORCE)
set(WALBERLA_ENABLE_SWEEPGEN   ON  CACHE BOOL "")
# OFF: use the active venv directly. ON: sweepgen builds its own venv from
# walberla/sweepgen-requirements.txt (may conflict with existing packages).
set(WALBERLA_SWEEPGEN_MANAGED_VENV OFF CACHE BOOL "" FORCE)

# GPU support follows TARGET_PLATFORM.
if(TARGET_PLATFORM STREQUAL "CPU")
    set(WALBERLA_BUILD_WITH_GPU_SUPPORT OFF CACHE BOOL "" FORCE)
    set(WALBERLA_BUILD_WITH_HIP         OFF CACHE BOOL "" FORCE)
    set(WALBERLA_BUILD_WITH_CUDA        OFF CACHE BOOL "" FORCE)
elseif(TARGET_PLATFORM STREQUAL "GPU-HIP")
    set(WALBERLA_BUILD_WITH_GPU_SUPPORT ON  CACHE BOOL "" FORCE)
    set(WALBERLA_BUILD_WITH_HIP         ON  CACHE BOOL "" FORCE)
    set(WALBERLA_BUILD_WITH_CUDA        OFF CACHE BOOL "" FORCE)
elseif(TARGET_PLATFORM STREQUAL "GPU-CUDA")
    set(WALBERLA_BUILD_WITH_GPU_SUPPORT ON  CACHE BOOL "" FORCE)
    set(WALBERLA_BUILD_WITH_HIP         OFF CACHE BOOL "" FORCE)
    set(WALBERLA_BUILD_WITH_CUDA        ON  CACHE BOOL "" FORCE)
endif()

# ============================================================================
# DEPENDENCIES
# ============================================================================
find_package(MPI REQUIRED)
if(TARGET_PLATFORM STREQUAL "GPU-HIP")
    find_package(hip REQUIRED)
elseif(TARGET_PLATFORM STREQUAL "GPU-CUDA")
    find_package(CUDA REQUIRED)
endif()

# ============================================================================
# WALBERLA SUBPROJECT (added once, shared by all apps)
# ============================================================================
add_subdirectory(walberla)
# Install MaterForge (this repo, one level up) editable into the sweepgen venv
# used during code generation, then populate the venv.
sweepgen_venv_require(-e ${CMAKE_SOURCE_DIR}/..)
sweepgen_venv_populate()

# ============================================================================
# APPS
# ============================================================================
if(BUILD_HEAT_EQUATION)
    add_subdirectory(HeatEquationKernel)
endif()
if(BUILD_COUETTE_FLOW)
    add_subdirectory(CouetteFlow)
endif()

message(STATUS "═══════════════════════════════════════════════════════════")
message(STATUS "MaterForge apps configured  (platform: ${TARGET_PLATFORM})")
message(STATUS "  HeatEquationKernel : ${BUILD_HEAT_EQUATION}")
message(STATUS "  CouetteFlow        : ${BUILD_COUETTE_FLOW}")
message(STATUS "═══════════════════════════════════════════════════════════")
