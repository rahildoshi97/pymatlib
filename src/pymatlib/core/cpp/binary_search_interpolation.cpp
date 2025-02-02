#include "include/binary_search_interpolation.h"
#include <iostream>
#include <cmath>


double interpolate_binary_search(
    const py::array_t<double>& temperature_array,
    double h_in,
    const py::array_t<double>& energy_density_array) {

    static constexpr double EPSILON = 1e-6;

    // Get array views
    auto temp_arr = temperature_array.unchecked<1>();
    auto energy_arr = energy_density_array.unchecked<1>();
    const size_t n = temp_arr.shape(0);

    // Input validation
    if (temperature_array.size() != energy_density_array.size() || n < 2) {
        throw std::runtime_error("Invalid array sizes");
    }

    // Determine array order and cache indices
    const bool is_ascending = temp_arr(0) < temp_arr(n-1);
    //std::cout << "C++ - is_ascending: " << is_ascending << std::endl;
    //std::cout << "C++ - h_in: " << h_in << std::endl;
    //std::cout << "C++ - First few T values: ";
    //for(size_t i = 0; i < 5; i++) std::cout << temp_arr(i) << " ";
    //std::cout << std::endl;
    //std::cout << "C++ - First few E values: ";
    //for(size_t i = 0; i < 5; i++) std::cout << energy_arr(i) << " ";
    //std::cout << std::endl;

    const size_t start_idx = is_ascending ? 0 : n-1;
    const size_t end_idx = is_ascending ? n-1 : 0;

    // Boundary checks
    if (h_in <= energy_arr(start_idx)) return temp_arr(start_idx);
    if (h_in >= energy_arr(end_idx)) return temp_arr(end_idx);

    // Binary search with optimized memory access
    size_t left = 0;
    size_t right = n - 1;

    while (left <= right) {
        const size_t mid = (left + right) / 2;
        const double mid_val = energy_arr(mid);
        //std::cout << "C++ - mid: " << mid << ", mid_val: " << mid_val << std::endl;

        if (std::abs(mid_val - h_in) < EPSILON) {
            //std::cout << "C++ - Exact match found at " << mid << std::endl;
            return temp_arr(mid);
        }

        const bool go_left = (mid_val > h_in) == is_ascending;
        if (go_left) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
        //std::cout << "C++ - left: " << left << ", right: " << right << std::endl;
    }

    //std::cout << "C++ - Final indices - left: " << left << ", right: " << right << std::endl;

    // Linear interpolation
    const double x0 = energy_arr(right);
    const double x1 = energy_arr(left);
    const double y0 = temp_arr(right);
    const double y1 = temp_arr(left);

    //std::cout << "C++ - Interpolation points:" << std::endl;
    //std::cout << "C++ - x0, x1: " << x0 << ", " << x1 << std::endl;
    //std::cout << "C++ - y0, y1: " << y0 << ", " << y1 << std::endl;

    double result = y0 + (y1 - y0) * (h_in - x0) / (x1 - x0);
    //std::cout << "C++ - Result: " << result << std::endl;
    return result;
}

/*
double interpolate_binary_search(
    const py::array_t<double>& temperature_array,
    double h_in,
    const py::array_t<double>& energy_density_array) {

    static constexpr double EPSILON = 1e-6;

    // Cache buffer requests and get pointers
    const auto temp_buf = temperature_array.request();
    const auto energy_buf = energy_density_array.request();

    // Use __restrict__ to inform compiler about non-aliasing
    const double* __restrict__ temp_ptr = static_cast<double*>(temp_buf.ptr);
    const double* __restrict__ energy_ptr = static_cast<double*>(energy_buf.ptr);
    const size_t n = temp_buf.size;

    // Input validation
    if (temp_buf.size != energy_buf.size || n < 2) {
        throw std::runtime_error("Invalid array sizes");
    }

    // Determine array order and cache indices
    const bool is_ascending = temp_ptr[0] < temp_ptr[n-1];
    const size_t start_idx = is_ascending ? 0 : n-1;
    const size_t end_idx = is_ascending ? n-1 : 0;
    std::cout << "C++ - is_ascending: " << is_ascending << std::endl;
    std::cout << "C++ - h_in: " << h_in << std::endl;
    std::cout << "C++ - First few T values: ";
    for(size_t i = 0; i < 5; i++) std::cout << temp_ptr[i] << " ";
    std::cout << std::endl;
    std::cout << "C++ - First few E values: ";
    for(size_t i = 0; i < 5; i++) std::cout << energy_ptr[i] << " ";
    std::cout << std::endl;

    if (energy_ptr[start_idx] >= energy_ptr[end_idx]) {
        throw std::runtime_error("Energy density must increase with temperature");
    }

    // Boundary checks
    if (h_in <= energy_ptr[start_idx]) return temp_ptr[start_idx];
    if (h_in >= energy_ptr[end_idx]) return temp_ptr[end_idx];

    // Binary search with optimized memory access
    size_t left = 0;
    size_t right = n - 1;

    while (left <= right) {
        const size_t mid = (left + right) / 2;
        const double mid_val = energy_ptr[mid];

        if (std::abs(mid_val - h_in) < EPSILON) {
            return temp_ptr[mid];
        }

        const bool go_left = (mid_val > h_in) == is_ascending;
        if (go_left) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    // Linear interpolation
    const size_t idx1 = is_ascending ? right : left;
    const size_t idx2 = is_ascending ? left : right;
    const double x0 = energy_ptr[idx1];
    const double dx = energy_ptr[idx2] - x0;
    const double y0 = temp_ptr[idx1];

    return y0 + (temp_ptr[idx2] - y0) * (h_in - x0) / dx;
}
*/
/*
double interpolate_binary_search(
    const py::array_t<double>& temperature_array,
    double h_in,
    const py::array_t<double>& energy_density_array) {

    static constexpr double EPSILON = 1e-6;

    const auto temp_buf = temperature_array.request();
    const auto energy_buf = energy_density_array.request();

    if (temp_buf.size != energy_buf.size) {
        throw std::runtime_error("Arrays must have same length");
    }

    const auto temp_ptr = static_cast<double*>(temp_buf.ptr);
    const auto energy_ptr = static_cast<double*>(energy_buf.ptr);
    const size_t n = temp_buf.size;

    if (n < 2) {
        throw std::runtime_error("Arrays must have at least 2 elements");
    }

    // Determine array order
    const bool is_ascending = temp_ptr[0] < temp_ptr[n-1];
    const size_t start_idx = is_ascending ? 0 : n-1;
    const size_t end_idx = is_ascending ? n-1 : 0;

    // Validate energy density increases with temperature
    if (energy_ptr[start_idx] >= energy_ptr[end_idx]) {
        throw std::runtime_error("Energy density must increase with temperature");
    }

    // Quick boundary checks
    if (h_in <= energy_ptr[start_idx]) return temp_ptr[start_idx];
    if (h_in >= energy_ptr[end_idx]) return temp_ptr[end_idx];

    // Binary search
    size_t left = 0;
    size_t right = n - 1;

    while (left <= right) {
        const size_t mid = (left + right) / 2;
        const double mid_val = energy_ptr[mid];

        if (std::abs(mid_val - h_in) < EPSILON) {
            return temp_ptr[mid];
        }

        if (mid_val > h_in) {
            if (is_ascending) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            if (is_ascending) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }

    // Linear interpolation
    const size_t idx1 = is_ascending ? right : left;
    const size_t idx2 = is_ascending ? left : right;

    const double x0 = energy_ptr[idx1];
    const double x1 = energy_ptr[idx2];
    const double y0 = temp_ptr[idx1];
    const double y1 = temp_ptr[idx2];

    const double slope = (y1 - y0) / (x1 - x0);
    return y0 + slope * (h_in - x0);
}
*/
/*
namespace detail {
    static constexpr double EPSILON = 1e-6;

    inline size_t interpolation_guess(const size_t left, const size_t right,
                                    const double* energy_ptr, const double h_in) {
        const double left_val = energy_ptr[left];
        const double right_val = energy_ptr[right];

        // Interpolation guess
        const double pos = static_cast<double>(left) +
            ((h_in - left_val) * static_cast<double>(right - left)) /
            (right_val - left_val);

        return static_cast<size_t>(std::round(pos));
    }
}

double interpolate_binary_search(
    const py::array_t<double>& temperature_array,
    double h_in,
    const py::array_t<double>& energy_density_array) {

    const auto temp_buf = temperature_array.request();
    const auto energy_buf = energy_density_array.request();

    if (temp_buf.size != energy_buf.size) {
        throw std::runtime_error("Arrays must have same length");
    }

    const double* temp_ptr = static_cast<double*>(temp_buf.ptr);
    const double* energy_ptr = static_cast<double*>(energy_buf.ptr);
    const size_t n = temp_buf.size;

    if (n < 2) {
        throw std::runtime_error("Arrays must have at least 2 elements");
    }

    // Determine array order
    const bool is_ascending = temp_ptr[0] < temp_ptr[n-1];
    const size_t start_idx = is_ascending ? 0 : n-1;
    const size_t end_idx = is_ascending ? n-1 : 0;

    // Validate energy density increases with temperature
    if (energy_ptr[start_idx] >= energy_ptr[end_idx]) {
        throw std::runtime_error("Energy density must increase with temperature");
    }

    // Quick boundary checks
    if (h_in <= energy_ptr[start_idx]) return temp_ptr[start_idx];
    if (h_in >= energy_ptr[end_idx]) return temp_ptr[end_idx];

    // Search with interpolation and binary fallback
    size_t left = 0;
    size_t right = n - 1;

    while (left <= right) {
        // Try interpolation search first
        size_t mid = detail::interpolation_guess(left, right, energy_ptr, h_in);

        // Fallback to binary search if interpolation guess is out of bounds
        if (mid < left || mid > right) {
            mid = left + (right - left) / 2;
        }

        const double mid_val = energy_ptr[mid];

        if (std::abs(mid_val - h_in) < detail::EPSILON) {
            return temp_ptr[mid];
        }

        if (mid_val > h_in) {
            if (is_ascending) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            if (is_ascending) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }

    // Linear interpolation
    const size_t idx1 = is_ascending ? right : left;
    const size_t idx2 = is_ascending ? left : right;

    const double x0 = energy_ptr[idx1];
    const double x1 = energy_ptr[idx2];
    const double y0 = temp_ptr[idx1];
    const double y1 = temp_ptr[idx2];

    const double slope = (y1 - y0) / (x1 - x0);
    return y0 + slope * (h_in - x0);
}
*/