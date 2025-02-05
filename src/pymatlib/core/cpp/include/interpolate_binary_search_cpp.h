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


template<typename Container>
double interpolate_binary_search_cpp(
    const Container& temperature_array,
    double h_in,
    const Container& energy_density_array) {

    static constexpr double EPSILON = 1e-6;

    // Input validation
    const size_t n = temperature_array.size();
    if (n != energy_density_array.size() || n < 2) {
        throw std::runtime_error("Invalid array sizes");
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

        if (std::abs(mid_val - h_in) < EPSILON) {
            return temperature_array[mid];
        }

        const bool go_left = (mid_val > h_in) == is_ascending;
        if (go_left) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    // Linear interpolation
    const double x0 = energy_density_array[right];
    const double x1 = energy_density_array[left];
    const double y0 = temperature_array[right];
    const double y1 = temperature_array[left];

    const double slope = (y1 - y0) / (x1 - x0);
    return y0 + slope * (h_in - x0);
}
