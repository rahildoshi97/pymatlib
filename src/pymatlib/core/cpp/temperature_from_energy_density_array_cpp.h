#pragma once
#include <cmath>
#include <vector>
#include <stdexcept>

/**
 * Fast temperature interpolation using C++ containers
 *
 * @param temperature_array Container of temperature values
 * @param h_in Energy density value to interpolate
 * @param energy_density_array Container of energy density values
 * @return Interpolated temperature value
 * @throws std::runtime_error if arrays have different lengths or wrong monotonicity
 */

namespace detail {
    static constexpr double EPSILON = 1e-6;
}

template<typename Container>
double temperature_from_energy_density_array_cpp(
    const Container& temperature_array,
    double h_in,
    const Container& energy_density_array) {

    // static constexpr double EPSILON = 1e-6;

    // Input validation
    const size_t n = temperature_array.size();
    if (n != energy_density_array.size()) {
        throw std::runtime_error("Arrays must have same length");
    }
    if (n < 2) {
        throw std::runtime_error("Arrays must have at least 2 elements");
    }

    // Determine array order
    const bool is_ascending = temperature_array[0] < temperature_array[n-1];
    const size_t start_idx = is_ascending ? 0 : n-1;
    const size_t end_idx = is_ascending ? n-1 : 0;

    // Validate energy density increases with temperature
    if (energy_density_array[start_idx] >= energy_density_array[end_idx]) {
        throw std::runtime_error("Energy density must increase with temperature");
    }

    // Quick boundary checks
    if (h_in <= energy_density_array[start_idx]) return temperature_array[start_idx];
    if (h_in >= energy_density_array[end_idx]) return temperature_array[end_idx];

    // Binary search
    size_t left = 0;
    size_t right = n - 1;

    while (left <= right) {
        const size_t mid = (left + right) / 2;
        const double mid_val = energy_density_array[mid];

        if (std::abs(mid_val - h_in) < detail::EPSILON) {
            return temperature_array[mid];
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

    const double x0 = energy_density_array[idx1];
    const double x1 = energy_density_array[idx2];
    const double y0 = temperature_array[idx1];
    const double y1 = temperature_array[idx2];

    const double slope = (y1 - y0) / (x1 - x0);
    return y0 + slope * (h_in - x0);
}
