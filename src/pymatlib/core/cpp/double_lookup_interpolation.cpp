#include "include/double_lookup_interpolation.h"
#include <cmath>


double interpolate_double_lookup(
    double E_target,
    const py::array_t<double>& T_eq,
    const py::array_t<double>& E_neq,
    const py::array_t<double>& E_eq,
    const py::array_t<int32_t>& idx_map) {

    // Cache all buffer requests at start
    const auto T_eq_buf = T_eq.request();
    const auto E_neq_buf = E_neq.request();
    const auto E_eq_buf = E_eq.request();
    const auto idx_map_buf = idx_map.request();

    // Get all pointers
    const double* __restrict__ T_eq_ptr = static_cast<double*>(T_eq_buf.ptr);
    const double* __restrict__ E_neq_ptr = static_cast<double*>(E_neq_buf.ptr);
    const double* __restrict__ E_eq_ptr = static_cast<double*>(E_eq_buf.ptr);
    const int32_t* __restrict__ idx_map_ptr = static_cast<int32_t*>(idx_map_buf.ptr);

    // Quick boundary checks
    if (E_target <= E_neq_ptr[0]) {
        return T_eq_ptr[0];
    }

    const size_t last_idx = E_neq_buf.size - 1;
    if (E_target >= E_neq_ptr[last_idx]) {
        return T_eq_ptr[last_idx];
    }

    // Cache delta_E to avoid recomputation
    const double delta_E = E_eq_ptr[1] - E_eq_ptr[0];
    const double E_eq_start = E_eq_ptr[0];

    // Compute indices
    const int idx_E_eq = std::min(
        static_cast<int>((E_target - E_eq_start) / delta_E),
        static_cast<int>(idx_map_buf.size - 1)
    );

    int idx_E_neq = idx_map_ptr[idx_E_eq];
    if (E_neq_ptr[idx_E_neq + 1] < E_target) {
        ++idx_E_neq;
    }

    // Linear interpolation with minimal temporaries
    const double E1 = E_neq_ptr[idx_E_neq];
    const double dE = E_neq_ptr[idx_E_neq + 1] - E1;
    const double T1 = T_eq_ptr[idx_E_neq];

    return T1 + (T_eq_ptr[idx_E_neq + 1] - T1) * (E_target - E1) / dE;
}

/*
double interpolate_double_lookup(
    double E_target,
    const py::array_t<double>& T_eq,
    const py::array_t<double>& E_neq,
    const py::array_t<double>& E_eq,
    const py::array_t<int32_t>& idx_map) {

    auto T_eq_buf = T_eq.request();
    auto E_neq_buf = E_neq.request();
    auto E_eq_buf = E_eq.request();
    auto idx_map_buf = idx_map.request();

    const double* T_eq_ptr = static_cast<double*>(T_eq_buf.ptr);
    const double* E_neq_ptr = static_cast<double*>(E_neq_buf.ptr);
    const double* E_eq_ptr = static_cast<double*>(E_eq_buf.ptr);
    const int32_t* idx_map_ptr = static_cast<int32_t*>(idx_map_buf.ptr);

    if (E_target <= E_neq_ptr[0]) {
        return T_eq_ptr[0];
    }
    if (E_target >= E_neq_ptr[E_neq_buf.size - 1]) {
        return T_eq_ptr[T_eq_buf.size - 1];
    }

    int idx_E_eq = static_cast<int>((E_target - E_eq_ptr[0]) / (E_eq_ptr[1] - E_eq_ptr[0]));
    idx_E_eq = std::min(idx_E_eq, static_cast<int>(idx_map_buf.size - 1));

    int idx_E_neq = idx_map_ptr[idx_E_eq];

    if (E_neq_ptr[idx_E_neq + 1] < E_target) {
        ++idx_E_neq;
    }

    double E1 = E_neq_ptr[idx_E_neq];
    double E2 = E_neq_ptr[idx_E_neq + 1];
    double T1 = T_eq_ptr[idx_E_neq];
    double T2 = T_eq_ptr[idx_E_neq + 1];

    return T1 + (T2 - T1) * (E_target - E1) / (E2 - E1);
}
*/
/*
double interpolate_double_lookup(
    double E_target,
    const py::array_t<double>& T_eq,
    const py::array_t<double>& E_neq,
    const py::array_t<double>& E_eq,
    const py::array_t<int32_t>& idx_map) {

    // Get buffer info once and cache sizes
    const auto& E_neq_buf = E_neq.request();
    const auto E_neq_size = E_neq_buf.size;
    const double* const E_neq_ptr = static_cast<double*>(E_neq_buf.ptr);

    // Quick boundary checks
    if (E_target <= E_neq_ptr[0]) {
        return static_cast<double*>(T_eq.request().ptr)[0];
    }
    if (E_target >= E_neq_ptr[E_neq_size - 1]) {
        return static_cast<double*>(T_eq.request().ptr)[E_neq_size - 1];
    }

    // Cache frequently used values
    const double* const E_eq_ptr = static_cast<double*>(E_eq.request().ptr);
    const double delta_E = E_eq_ptr[1] - E_eq_ptr[0];

    // Calculate index in equidistant grid
    const int idx_E_eq = std::min(
        static_cast<int>((E_target - E_eq_ptr[0]) / delta_E),
        static_cast<int>(idx_map.request().size - 1)
    );

    // Get mapped index and data pointers
    int idx_E_neq = static_cast<int32_t*>(idx_map.request().ptr)[idx_E_eq];
    const double* const T_eq_ptr = static_cast<double*>(T_eq.request().ptr);

    // Adjust index if needed
    if (E_neq_ptr[idx_E_neq + 1] < E_target) {
        ++idx_E_neq;
    }

    // Linear interpolation using direct array access
    const double E1 = E_neq_ptr[idx_E_neq];
    const double E2 = E_neq_ptr[idx_E_neq + 1];
    const double T1 = T_eq_ptr[idx_E_neq];
    const double T2 = T_eq_ptr[idx_E_neq + 1];

    return T1 + ((T2 - T1) * (E_target - E1)) / (E2 - E1);
}
*/


