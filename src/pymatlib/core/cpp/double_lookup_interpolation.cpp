#include "include/double_lookup_interpolation.h"
#include <iostream>
#include <cmath>


double interpolate_double_lookup(
    double E_target,
    const py::array_t<double>& T_eq,
    const py::array_t<double>& E_neq,
    const py::array_t<double>& E_eq,
    double inv_delta_E_eq,
    const py::array_t<int32_t>& idx_map) {

    // Get array views with bounds checking disabled for performance
    auto T_eq_arr = T_eq.unchecked<1>();
    auto E_neq_arr = E_neq.unchecked<1>();
    auto E_eq_arr = E_eq.unchecked<1>();
    auto idx_map_arr = idx_map.unchecked<1>();

    // Cache array size
    const size_t n = T_eq_arr.shape(0);

    // Quick boundary checks with cached values
    if (E_target <= E_neq_arr(0)) {
        return T_eq_arr(0);
    }

    if (E_target >= E_neq_arr(n-1)) {
        return T_eq_arr(n-1);
    }

    const int idx_E_eq = static_cast<int>((E_target - E_eq_arr(0)) * inv_delta_E_eq);

    int idx_E_neq = idx_map_arr(idx_E_eq);

    idx_E_neq += E_neq_arr(idx_E_neq + 1) < E_target;

    // Get interpolation index
    const double E1 = E_neq_arr(idx_E_neq);
    const double E2 = E_neq_arr(idx_E_neq + 1);
    const double T1 = T_eq_arr(idx_E_neq);
    const double T2 = T_eq_arr(idx_E_neq + 1);

    // Optimized linear interpolation
    const double slope = (T2 - T1) / (E2 - E1);
    return T1 + slope * (E_target - E1);
}
