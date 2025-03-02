cmake_minimum_required(VERSION 3.10)
project(fast_interpolation)

# Find required packages
find_package(Python COMPONENTS Interpreter Development NumPy REQUIRED)
execute_process(
        COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 PATHS ${PYBIND11_CMAKE_DIR} NO_DEFAULT_PATH REQUIRED)

# Create interface library for header-only template implementation
add_library(interpolation_template INTERFACE)
target_include_directories(interpolation_template INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add the Pybind11 module
pybind11_add_module(fast_interpolation
        module.cpp
        binary_search_interpolation.cpp
        double_lookup_interpolation.cpp
)
target_include_directories(fast_interpolation PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_features(fast_interpolation PRIVATE cxx_std_11)
target_compile_options(fast_interpolation PRIVATE -O3)
target_link_libraries(fast_interpolation PRIVATE
        interpolation_template
        pybind11::module
        Python::NumPy
)
