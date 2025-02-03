#include "include/binary_search_interpolation.h"
#include <iostream>
#include <cmath>


double interpolate_binary_search(
    const py::array_t<double>& temperature_array,
    double h_in,
    const py::array_t<double>& energy_density_array) {

    static constexpr double EPSILON = 1e-6;

    // Get array views
    auto temp_arr = temperature_array.unchecked<1>();
    auto energy_arr = energy_density_array.unchecked<1>();
    const size_t n = temp_arr.shape(0);

    // Input validation
    if (temperature_array.size() != energy_density_array.size() || n < 2) {
        throw std::runtime_error("Invalid array sizes");
    }

    // Determine array order and cache indices
    const bool is_ascending = temp_arr(0) < temp_arr(n-1);

    const size_t start_idx = is_ascending ? 0 : n-1;
    const size_t end_idx = is_ascending ? n-1 : 0;

    // Boundary checks
    if (h_in <= energy_arr(start_idx)) return temp_arr(start_idx);
    if (h_in >= energy_arr(end_idx)) return temp_arr(end_idx);

    // Binary search with optimized memory access
    size_t left = 0;
    size_t right = n - 1;

    while (left <= right) {
        const size_t mid = (left + right) / 2;
        const double mid_val = energy_arr(mid);

        if (std::abs(mid_val - h_in) < EPSILON) {
            return temp_arr(mid);
        }

        const bool go_left = (mid_val > h_in) == is_ascending;
        if (go_left) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }


    // Linear interpolation
    const double x0 = energy_arr(right);
    const double x1 = energy_arr(left);
    const double y0 = temp_arr(right);
    const double y1 = temp_arr(left);

    return y0 + (y1 - y0) * (h_in - x0) / (x1 - x0);
}
