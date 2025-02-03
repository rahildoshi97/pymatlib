#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "interpolate_binary_search_cpp.h"

namespace py = pybind11;

/**
 * Fast temperature interpolation using interpolation search with binary fallback
 *
 * @param temperature_array Array of temperature values (monotonically decreasing)
 * @param h_in Energy density value to interpolate
 * @param energy_density_array Array of energy density values (monotonically decreasing)
 * @return Interpolated temperature value
 * @throws std::runtime_error if arrays have different lengths or wrong monotonicity
 */
double interpolate_binary_search(
    const py::array_t<double>& temperature_array,
    double h_in,
    const py::array_t<double>& energy_density_array);
