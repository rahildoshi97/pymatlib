#include "temperature_from_energy_density_array.h"
#include <cmath>


double temperature_from_energy_density_array(
    const py::array_t<double>& temperature_array,
    double h_in,
    const py::array_t<double>& energy_density_array) {

    static constexpr double EPSILON = 1e-6;

    const auto temp_buf = temperature_array.request();
    const auto energy_buf = energy_density_array.request();

    if (temp_buf.size != energy_buf.size) {
        throw std::runtime_error("Arrays must have same length");
    }

    const auto temp_ptr = static_cast<double*>(temp_buf.ptr);
    const auto energy_ptr = static_cast<double*>(energy_buf.ptr);
    const size_t n = temp_buf.size;

    if (n < 2) {
        throw std::runtime_error("Arrays must have at least 2 elements");
    }

    // Determine array order
    const bool is_ascending = temp_ptr[0] < temp_ptr[n-1];
    const size_t start_idx = is_ascending ? 0 : n-1;
    const size_t end_idx = is_ascending ? n-1 : 0;

    // Validate energy density increases with temperature
    /*if (energy_ptr[start_idx] >= energy_ptr[end_idx]) {
        throw std::runtime_error("Energy density must increase with temperature");
    }*/

    // Quick boundary checks
    if (h_in <= energy_ptr[start_idx]) return temp_ptr[start_idx];
    if (h_in >= energy_ptr[end_idx]) return temp_ptr[end_idx];

    // Binary search
    size_t left = 0;
    size_t right = n - 1;

    while (left <= right) {
        const size_t mid = (left + right) / 2;
        const double mid_val = energy_ptr[mid];

        if (std::abs(mid_val - h_in) < EPSILON) {
            return temp_ptr[mid];
        }

        if (mid_val > h_in) {
            if (is_ascending) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            if (is_ascending) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }

    // Linear interpolation
    const size_t idx1 = is_ascending ? right : left;
    const size_t idx2 = is_ascending ? left : right;

    const double x0 = energy_ptr[idx1];
    const double x1 = energy_ptr[idx2];
    const double y0 = temp_ptr[idx1];
    const double y1 = temp_ptr[idx2];

    const double slope = (y1 - y0) / (x1 - x0);
    return y0 + slope * (h_in - x0);
}

PYBIND11_MODULE(fast_interpolation, m) {
    m.doc() = "Fast temperature interpolation using binary search";
    m.def("temperature_from_energy_density_array",
          &temperature_from_energy_density_array,
          "Find temperature using binary search and linear interpolation",
          py::arg("temperature_array"),
          py::arg("h_in"),
          py::arg("energy_density_array"));
}
