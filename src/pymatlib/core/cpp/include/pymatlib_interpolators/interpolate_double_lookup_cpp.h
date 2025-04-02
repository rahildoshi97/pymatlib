#pragma once
#include <cmath>
#include <stdexcept>


template<typename ArrayContainer>
double interpolate_double_lookup_cpp(
    double y_target,
    const ArrayContainer& arrs) {

    // Handle boundary cases
    if (y_target <= arrs.y_neq[0]) return arrs.x_eq[0];
    if (y_target >= arrs.y_neq.back()) return arrs.x_eq.back();

    // Calculate index in equidistant grid
    int idx_y_eq = static_cast<int>((y_target - arrs.y_eq[0]) * arrs.inv_delta_y_eq);
    idx_y_eq = std::min(idx_y_eq, static_cast<int>(arrs.idx_map.size() - 1));

    // Get index from mapping
    int idx_y_neq = arrs.idx_map[idx_y_eq];

    // Make sure we don't go out of bounds
    idx_y_neq = std::min(idx_y_neq, static_cast<int>(arrs.y_neq.size() - 2));

    // Adjust index if needed
    idx_y_neq += arrs.y_neq[idx_y_neq + 1] < y_target;

    // Get interpolation points
    const double x1 = arrs.x_eq[idx_y_neq];
    const double x2 = arrs.x_eq[idx_y_neq + 1];
    const double y1 = arrs.y_neq[idx_y_neq];
    const double y2 = arrs.y_neq[idx_y_neq + 1];

    // Linear interpolation
    const double inv_slope = (x2 - x1) / (y2 - y1);
    return x1 + inv_slope * (y_target - y1);
}
