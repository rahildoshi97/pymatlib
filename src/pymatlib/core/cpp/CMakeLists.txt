cmake_minimum_required(VERSION 3.10)
project(pymatlib_interpolators)

# Find required packages
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)

# Create interface library for header-only template implementation
add_library(pymatlib_interpolators INTERFACE)
target_include_directories(pymatlib_interpolators INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
