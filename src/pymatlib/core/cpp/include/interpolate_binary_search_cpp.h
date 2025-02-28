#pragma once
#include <cmath>
#include <stdexcept>


template<typename ArrayContainer>
double interpolate_binary_search_cpp(
    double E_target,
    const ArrayContainer& arrs) {

    static constexpr double EPSILON = 1e-6;
    const size_t n = arrs.T_bs.size();

    // Quick boundary checks
    if (E_target <= arrs.E_bs[0]) return arrs.T_bs[0];
    if (E_target >= arrs.E_bs.back()) return arrs.T_bs.back();

    // Binary search
    size_t left = 0;
    size_t right = n - 1;

    while (right - left > 1) {
        const size_t mid = left + (right - left) / 2;

        if (std::abs(arrs.E_bs[mid] - E_target) < EPSILON) {
            return arrs.T_bs[mid];
        }

        if (arrs.E_bs[mid] > E_target) {
            right = mid;
        } else {
            left = mid;
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
