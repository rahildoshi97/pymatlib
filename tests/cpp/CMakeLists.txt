cmake_minimum_required(VERSION 3.10)
project(pymatlib_tests)

# Find required packages
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
execute_process(
        COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 PATHS ${PYBIND11_CMAKE_DIR} NO_DEFAULT_PATH REQUIRED)

# Get the path to the source directory
set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../src/pymatlib/core/cpp")

# Create interface library for header-only template implementation
add_library(interpolation_template INTERFACE)
target_include_directories(interpolation_template INTERFACE
        ${SRC_DIR}
        ${SRC_DIR}/include
)

# Add C++ test executable
add_executable(fast_interpolation_test_cpp test_interpolation.cpp)
target_include_directories(fast_interpolation_test_cpp PRIVATE
        ${SRC_DIR}
        ${SRC_DIR}/include
)
target_compile_features(fast_interpolation_test_cpp PRIVATE cxx_std_11)
target_compile_options(fast_interpolation_test_cpp PRIVATE -O3)
target_link_libraries(fast_interpolation_test_cpp PRIVATE
        interpolation_template
)

# Optional: Add custom target for running Python tests if you have a Python test file
# add_custom_target(fast_interpolation_test_py
#         COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_interpolation.py
#         DEPENDS fast_interpolation_test_cpp
#         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#         COMMENT "Running Python tests"
# )
