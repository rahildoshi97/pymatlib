#ifndef DOUBLE_LOOKUP_INTERPOLATION_H
#define DOUBLE_LOOKUP_INTERPOLATION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "interpolate_double_lookup_cpp.h"

namespace py = pybind11;

double interpolate_double_lookup(
    double E_target,
    const py::array_t<double>& T_eq,
    const py::array_t<double>& E_neq,
    const py::array_t<double>& E_eq,
    double inv_delta_E_eq,
    const py::array_t<int32_t>& idx_map);

#endif // DOUBLE_LOOKUP_INTERPOLATION_H
