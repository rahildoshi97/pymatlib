cmake_minimum_required(VERSION 3.10)
project(fast_interpolation)

find_package(Python COMPONENTS Development NumPy REQUIRED)

execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir OUTPUT_VARIABLE PYBIND11_CMAKE_PATH)
string(STRIP "${PYBIND11_CMAKE_PATH}" PYBIND11_CMAKE_PATH)
find_package(pybind11 PATHS "${PYBIND11_CMAKE_PATH}" NO_DEFAULT_PATH REQUIRED)

pybind11_add_module(fast_interpolation temperature_from_energy_density_array.cpp)
target_compile_features(fast_interpolation PRIVATE cxx_std_11)
target_compile_options(fast_interpolation PRIVATE -O3)
target_link_libraries(fast_interpolation PRIVATE Python::NumPy)
