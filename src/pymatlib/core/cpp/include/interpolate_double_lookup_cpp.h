#pragma once
#include <cmath>
#include <vector>
#include <stdexcept>


template<typename Container>
double interpolate_double_lookup_cpp(
    double E_target,
    const Container& T_eq,
    const Container& E_neq,
    const Container& E_eq,
    const Container& idx_map) {

    // Handle boundary cases
    if (E_target <= E_neq[0]) {
        return T_eq[0];
    }
    if (E_target >= E_neq.back()) {
        return T_eq.back();
    }

    // Calculate index in equidistant grid
    int idx_E_eq = static_cast<int>((E_target - E_eq[0]) / (E_eq[1] - E_eq[0]));
    idx_E_eq = std::min(idx_E_eq, static_cast<int>(idx_map.size() - 1));

    // Get index from mapping
    int idx_E_neq = idx_map[idx_E_eq];

    // Adjust index if needed
    if (E_neq[idx_E_neq + 1] < E_target) {
        ++idx_E_neq;
    }

    // Get interpolation points
    const double E1 = E_neq[idx_E_neq];
    const double E2 = E_neq[idx_E_neq + 1];
    const double T1 = T_eq[idx_E_neq];
    const double T2 = T_eq[idx_E_neq + 1];

    // Linear interpolation
    return T1 + (T2 - T1) * (E_target - E1) / (E2 - E1);
}
