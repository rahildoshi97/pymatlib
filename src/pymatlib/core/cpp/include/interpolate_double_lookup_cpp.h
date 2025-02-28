#pragma once
#include <cmath>
#include <stdexcept>


template<typename ArrayContainer>
double interpolate_double_lookup_cpp(
    double E_target,
    const ArrayContainer& arrs) {

    const size_t n = arrs.T_eq.size();
    if (n != arrs.E_neq.size() || n < 2) {
        throw std::runtime_error("Invalid array sizes");
    }

    // Determine array order
    const bool is_ascending = arrs.T_eq[0] < arrs.T_eq[n-1];
    const size_t start_idx = is_ascending ? 0 : n-1;
    const size_t end_idx = is_ascending ? n-1 : 0;

    // Validate energy density increases with temperature
    if (arrs.E_neq[start_idx] >= arrs.E_neq[end_idx]) {
        throw std::runtime_error("Energy density must increase with temperature");
    }

    // Handle boundary cases
    if (E_target <= arrs.E_neq[0]) return arrs.T_eq[0];
    if (E_target >= arrs.E_neq.back()) return arrs.T_eq.back();

    // Calculate index in equidistant grid
    int idx_E_eq = static_cast<int>((E_target - arrs.E_eq[0]) * arrs.inv_delta_E_eq);
    idx_E_eq = std::min(idx_E_eq, static_cast<int>(arrs.idx_map.size() - 1));

    // Get index from mapping
    int idx_E_neq = arrs.idx_map[idx_E_eq];

    // Adjust index if needed
    idx_E_neq += arrs.E_neq[idx_E_neq + 1] < E_target;

    // Get interpolation points
    const double E1 = arrs.E_neq[idx_E_neq];
    const double E2 = arrs.E_neq[idx_E_neq + 1];
    const double T1 = arrs.T_eq[idx_E_neq];
    const double T2 = arrs.T_eq[idx_E_neq + 1];

    // Linear interpolation
    const double slope = (T2 - T1) / (E2 - E1);
    return T1 + slope * (E_target - E1);
}
