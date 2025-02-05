#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "include/binary_search_interpolation.h"
#include "include/double_lookup_interpolation.h"

namespace py = pybind11;

PYBIND11_MODULE(fast_interpolation, m) {
    m.doc() = "Fast interpolation methods implementation";

    m.def("interpolate_binary_search",
          &interpolate_binary_search,
          "Find temperature using binary search and linear interpolation",
          py::arg("temperature_array"),
          py::arg("h_in"),
          py::arg("energy_density_array"));

    m.def("interpolate_double_lookup",
          &interpolate_double_lookup,
          "Fast interpolation using double lookup method and linear interpolation",
          py::arg("E_target"),
          py::arg("T_eq"),
          py::arg("E_neq"),
          py::arg("E_eq"),
          py::arg("inv_delta_E_eq"),
          py::arg("idx_map"));
}
