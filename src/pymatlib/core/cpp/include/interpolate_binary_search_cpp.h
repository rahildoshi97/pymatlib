#pragma once
#include <cmath>
#include <stdexcept>


template<typename ArrayContainer>
double interpolate_binary_search_cpp(
    double E_target,
    const ArrayContainer& arrs) {

    static constexpr double EPSILON = 1e-6;

    // Input validation
    const size_t n = arrs.T_bs.size();
    if (n != arrs.E_bs.size() || n < 2) {
        throw std::runtime_error("Invalid array sizes");
    }

    // Determine array order
    const bool is_ascending = arrs.T_bs[0] < arrs.T_bs[n-1];
    const size_t start_idx = is_ascending ? 0 : n-1;
    const size_t end_idx = is_ascending ? n-1 : 0;

    // Validate energy density increases with temperature
    if (arrs.E_bs[start_idx] >= arrs.E_bs[end_idx]) {
        throw std::runtime_error("Energy density must increase with temperature");
    }

    // Quick boundary checks
    if (E_target <= arrs.E_bs[start_idx]) return arrs.T_bs[start_idx];
    if (E_target >= arrs.E_bs[end_idx]) return arrs.T_bs[end_idx];

    // Binary search
    size_t left = 0;
    size_t right = n - 1;

    while (left <= right) {
        const size_t mid = (left + right) / 2;
        const double mid_val = arrs.E_bs[mid];

        if (std::abs(mid_val - E_target) < EPSILON) {
            return arrs.T_bs[mid];
        }

        const bool go_left = (mid_val > E_target) == is_ascending;
        if (go_left) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    // Linear interpolation
    const double x0 = arrs.E_bs[right];
    const double x1 = arrs.E_bs[left];
    const double y0 = arrs.T_bs[right];
    const double y1 = arrs.T_bs[left];

    const double slope = (y1 - y0) / (x1 - x0);
    return y0 + slope * (E_target - x0);
}
