#include "include/double_lookup_interpolation.h"
#include <iostream>
#include <cmath>


double interpolate_double_lookup(
    double E_target,
    const py::array_t<double>& T_eq,
    const py::array_t<double>& E_neq,
    const py::array_t<double>& E_eq,
    const py::array_t<int32_t>& idx_map) {

    // Get array views with bounds checking disabled for performance
    auto T_eq_arr = T_eq.unchecked<1>();
    auto E_neq_arr = E_neq.unchecked<1>();
    auto E_eq_arr = E_eq.unchecked<1>();
    auto idx_map_arr = idx_map.unchecked<1>();

    // Cache frequently accessed values
    const size_t last_idx = E_neq_arr.shape(0) - 1;
    const double first_e = E_neq_arr(0);
    const double last_e = E_neq_arr(last_idx);

    // Quick boundary checks with cached values
    if (E_target <= first_e) {
        return T_eq_arr(0);
    }

    if (E_target >= last_e) {
        return T_eq_arr(last_idx);
    }

    // Precompute and cache interpolation parameters
    const double E_eq_start = E_eq_arr(0);
    const double delta_E = E_eq_arr(1) - E_eq_start;
    const double inv_delta_E = 1.0 / delta_E;

    const int idx_E_eq = std::min(
        static_cast<int>((E_target - E_eq_start) * inv_delta_E),
        static_cast<int>(idx_map_arr.shape(0) - 1)
    );
    //std::cout << "idx_E_eq: " << idx_E_eq << std::endl;

    int idx_E_neq = idx_map_arr(idx_E_eq);

    if (E_neq_arr(idx_E_neq + 1) < E_target) {
        ++idx_E_neq;
    }

    // Get interpolation index
    const double E1 = E_neq_arr(idx_E_neq);
    const double E2 = E_neq_arr(idx_E_neq + 1);
    const double T1 = T_eq_arr(idx_E_neq);
    const double T2 = T_eq_arr(idx_E_neq + 1);

    // Optimized linear interpolation
    const double inv_dE = 1.0 / (E2 - E1);

    return T1 + (T2 - T1) * (E_target - E1) * inv_dE;
}
