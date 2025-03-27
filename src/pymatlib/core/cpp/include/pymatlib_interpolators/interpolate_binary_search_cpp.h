#pragma once
#include <cmath>
#include <stdexcept>


template<typename ArrayContainer>
double interpolate_binary_search_cpp(
    double y_target,
    const ArrayContainer& arrs) {

    static constexpr double EPSILON = 1e-6;
    const size_t n = arrs.x_bs.size();

    // Quick boundary checks
    if (y_target <= arrs.y_bs[0]) return arrs.x_bs[0];
    if (y_target >= arrs.y_bs.back()) return arrs.x_bs.back();

    // Binary search
    size_t left = 0;
    size_t right = n - 1;

    while (right - left > 1) {
        const size_t mid = left + (right - left) / 2;

        if (std::abs(arrs.y_bs[mid] - y_target) < EPSILON) {
            return arrs.x_bs[mid];
        }

        if (arrs.y_bs[mid] > y_target) {
            right = mid;
        } else {
            left = mid;
        }
    }

    // Linear interpolation
    const double x1 = arrs.y_bs[right];
    const double x2 = arrs.y_bs[left];
    const double y1 = arrs.x_bs[right];
    const double y2 = arrs.x_bs[left];

    const double slope = (y2 - y1) / (x2 - x1);
    return y1 + slope * (y_target - x1);
}
