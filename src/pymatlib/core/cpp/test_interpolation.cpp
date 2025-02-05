#include "interpolate_binary_search_cpp.h"
#include "interpolate_double_lookup_cpp.h"
#include <array>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <iostream>


// Helper function to compare floating point numbers
bool is_equal(double a, double b, double tolerance = 1e-10) {
    return std::abs(a - b) < tolerance;
}


void test_basic_functionality() {
    std::cout << "\nTesting basic functionality..." << std::endl;

    // Setup test data with precise values
    // Ascending order vectors
    const std::vector<double> T_vec = {3243.15, 3253.15, 3263.15, 3273.15};
    const std::vector<double> E_vec = {1.68e10, 1.69e10, 1.70e10, 1.71e10};
    // Ascending order arrays
    const std::array<double, 4> T_arr = {3243.15, 3253.15, 3263.15, 3273.15};
    const std::array<double, 4> E_arr = {1.68e10, 1.69e10, 1.70e10, 1.71e10};

    // Test middle point interpolation
    const double test_E = 1.695e10;
    const double expected_T = 3258.15;

    // Binary Search Tests
    {
        double result_vec = interpolate_binary_search_cpp(T_vec, test_E, E_vec);
        assert(is_equal(result_vec, expected_T));

        double result_arr = interpolate_binary_search_cpp(T_arr, test_E, E_arr);
        assert(is_equal(result_arr, expected_T));

        // Verify vector and array results match
        assert(is_equal(result_vec, result_arr));
    }

    // Double Lookup Tests
    {
        // Setup double lookup specific data
        std::vector<double> E_eq_vec = {1.68e10, 1.69e10, 1.70e10, 1.71e10};
        std::array<double, 4> E_eq_arr = {1.68e10, 1.69e10, 1.70e10, 1.71e10};
        std::vector<double> idx_map_vec = {0, 1, 2, 3};
        std::array<double, 4> idx_map_arr = {0, 1, 2, 3};
        const double inv_delta_E_eq = 1.0 / (1e9);

        double result_vec = interpolate_double_lookup_cpp(
            test_E, T_vec, E_vec, E_eq_vec, inv_delta_E_eq, idx_map_vec);
        assert(is_equal(result_vec, expected_T));

        double result_arr = interpolate_double_lookup_cpp(
            test_E, T_arr, E_arr, E_eq_arr, inv_delta_E_eq, idx_map_arr);
        assert(is_equal(result_arr, expected_T));

        // Verify vector and array results match
        assert(is_equal(result_vec, result_arr));
    }
}


void test_edge_cases_and_errors() {
    std::cout << "Testing edge cases and error handling..." << std::endl;

    // Test data for vectors
    const std::vector<double> T_vec = {3243.15, 3253.15, 3263.15, 3273.15};
    const std::vector<double> E_vec = {1.68e10, 1.69e10, 1.70e10, 1.71e10};

    // Test data for arrays
    const std::array<double, 4> T_arr = {3243.15, 3253.15, 3263.15, 3273.15};
    const std::array<double, 4> E_arr = {1.68e10, 1.69e10, 1.70e10, 1.71e10};

    // Test boundary values
    {
        // Vector tests
        double result_vec_min = interpolate_binary_search_cpp(T_vec, 1.67e10, E_vec);
        double result_vec_max = interpolate_binary_search_cpp(T_vec, 1.72e10, E_vec);
        assert(is_equal(result_vec_min, 3243.15));
        assert(is_equal(result_vec_max, 3273.15));

        // Array tests
        double result_arr_min = interpolate_binary_search_cpp(T_arr, 1.67e10, E_arr);
        double result_arr_max = interpolate_binary_search_cpp(T_arr, 1.72e10, E_arr);
        assert(is_equal(result_arr_min, 3243.15));
        assert(is_equal(result_arr_max, 3273.15));
    }

    // Error cases
    {
        // Test vector size mismatch
        std::vector<double> wrong_size_vec = {1.0};
        bool caught_vector_error = false;
        try {
            interpolate_binary_search_cpp(T_vec, 1.70e10, wrong_size_vec);
        } catch (const std::runtime_error&) {
            caught_vector_error = true;
        }
        assert(caught_vector_error);

        // For arrays, test with non-monotonic values
        std::array<double, 4> non_monotonic_arr = {1.71e10, 1.69e10, 1.70e10, 1.68e10};  // Values not in order
        bool caught_array_error = false;
        try {
            interpolate_binary_search_cpp(T_arr, 1.70e10, non_monotonic_arr);
        } catch (const std::runtime_error&) {
            caught_array_error = true;
        }
        assert(caught_array_error);

        // Compile-time size check for arrays
        static_assert(T_arr.size() >= 2, "Array size must be at least 2");
        static_assert(E_arr.size() >= 2, "Array size must be at least 2");
    }

    // Double Lookup edge cases
    {
        std::vector<double> E_eq_vec = {1.68e10, 1.69e10, 1.70e10, 1.71e10};
        std::vector<double> idx_map_vec = {0, 1, 2, 3};
        std::array<double, 4> E_eq_arr = {1.68e10, 1.69e10, 1.70e10, 1.71e10};
        std::array<double, 4> idx_map_arr = {0, 1, 2, 3};
        const double inv_delta_E_eq = 1.0 / (1e9);

        // Vector tests
        double result_vec_min = interpolate_double_lookup_cpp(
            1.67e10, T_vec, E_vec, E_eq_vec, inv_delta_E_eq, idx_map_vec);
        assert(is_equal(result_vec_min, 3243.15));

        // Array tests
        double result_arr_min = interpolate_double_lookup_cpp(
            1.67e10, T_arr, E_arr, E_eq_arr, inv_delta_E_eq, idx_map_arr);
        assert(is_equal(result_arr_min, 3243.15));
    }
}


void test_interpolation_accuracy() {
    std::cout << "\nTesting interpolation accuracy..." << std::endl;

    // Test data with precise values
    struct TestCase {
        double input_E;
        double expected_T;
        double tolerance;
        std::string description;
    };

    const std::vector<TestCase> test_cases = {
        // For E=1.705e10 (between 1.70e10 and 1.71e10)
        {1.705e10, 3268.15, 1e-10, "Mid-point interpolation"},
        // For E=1.695e10 (between 1.69e10 and 1.70e10)
        {1.695e10, 3258.15, 1e-10, "Quarter-point interpolation"},
        // For E=1.685e10 (between 1.68e10 and 1.69e10)
        {1.685e10, 3248.15, 1e-10, "Eighth-point interpolation"},
        // Near exact points
        {1.70e10, 3263.15, 1e-8, "Near-exact point"},
        {1.69e10, 3253.15, 1e-8, "Near-exact point"}
    };

    // Setup arrays with high precision values
    // For std::vector
    const std::vector<double> T_vec = {
        3243.15000000, 3253.15000000, 3263.15000000, 3273.15000000
    };
    const std::vector<double> E_vec = {
        1.68000000e10, 1.69000000e10, 1.70000000e10, 1.71000000e10
    };

    // For std::array
    const std::array<double, 4> T_arr = {
        3243.15000000, 3253.15000000, 3263.15000000, 3273.15000000
    };
    const std::array<double, 4> E_arr = {
        1.68000000e10, 1.69000000e10, 1.70000000e10, 1.71000000e10
    };

    // Setup for double lookup
    std::vector<double> E_eq_vec = {
        1.68000000e10, 1.69000000e10, 1.70000000e10, 1.71000000e10
    };
    std::array<double, 4> E_eq_arr = {
        1.68000000e10, 1.69000000e10, 1.70000000e10, 1.71000000e10
    };
    std::vector<double> idx_map_vec = {0, 1, 2, 3};
    std::array<double, 4> idx_map_arr = {0, 1, 2, 3};
    const double inv_delta_E_eq = 1.0 / (1e9);

    for (const auto& test : test_cases) {
        // Binary Search Tests
        double result_bin_vec = interpolate_binary_search_cpp(
            T_vec, test.input_E, E_vec);
        double result_bin_arr = interpolate_binary_search_cpp(
            T_arr, test.input_E, E_arr);

        assert(is_equal(result_bin_vec, test.expected_T, test.tolerance));
        assert(is_equal(result_bin_arr, test.expected_T, test.tolerance));
        assert(is_equal(result_bin_vec, result_bin_arr, test.tolerance));

        // Double Lookup Tests
        double result_dl_vec = interpolate_double_lookup_cpp(
            test.input_E, T_vec, E_vec, E_eq_vec, inv_delta_E_eq, idx_map_vec);
        double result_dl_arr = interpolate_double_lookup_cpp(
            test.input_E, T_arr, E_arr, E_eq_arr, inv_delta_E_eq, idx_map_arr);

        assert(is_equal(result_dl_vec, test.expected_T, test.tolerance));
        assert(is_equal(result_dl_arr, test.expected_T, test.tolerance));
        assert(is_equal(result_dl_vec, result_dl_arr, test.tolerance));
    }
}


void test_performance() {
    std::cout << "\nTesting performance..." << std::endl;

    // Test with different sizes
    const std::vector<size_t> test_sizes = {100, 1000, 10000, 100000};

    for (size_t size : test_sizes) {
        std::cout << "\nTesting with array size: " << size << std::endl;

        // Generate test data
        std::vector<double> T_vec(size);
        std::vector<double> E_vec(size);
        std::vector<double> E_eq_vec(size);
        std::vector<double> idx_map_vec(size);

        // Fill with realistic values
        for (size_t i = 0; i < size; ++i) {
            T_vec[i] = 3273.15 + i * 0.01;
            E_vec[i] = 1.71e10 + i * 1e6 + 1e3 * i/2.;
            E_eq_vec[i] = 1.71e10 + i * 1e6 + 1e3 * i/2.;
            idx_map_vec[i] = i;
        }

        // Create smaller arrays for comparison
        constexpr size_t arr_size = 1000;
        std::array<double, arr_size> T_arr;
        std::array<double, arr_size> E_arr;
        std::array<double, arr_size> E_eq_arr;
        std::array<double, arr_size> idx_map_arr;

        for (size_t i = 0; i < arr_size; ++i) {
            T_arr[i] = T_vec[i];
            E_arr[i] = E_vec[i];
            E_eq_arr[i] = E_eq_vec[i];
            idx_map_arr[i] = idx_map_vec[i];
        }

        const double inv_delta_E_eq = 1.0 / (1e6);
        const int num_tests = 1000;

        // Test points
        /*std::vector<double> test_points(num_tests);
        for (int i = 0; i < num_tests; ++i) {
            test_points[i] = 1.70e10 + (i * 1e6);
        }*/

        // Generate random monotonically increasing test points
        std::vector<double> test_points(num_tests);
        test_points[0] = 1.70e10;  // Start value
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(1e5, 1e6);  // Random increments between 1e5 and 1e6

        for (int i = 1; i < num_tests; ++i) {
            test_points[i] = test_points[i-1] + dist(gen);
        }

        // Measure binary search performance
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (const double& E : test_points) {
                volatile double result = interpolate_binary_search_cpp(
                    T_vec, E, E_vec);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count();
            std::cout << "Binary Search (vector) time: " << duration << " ns" << std::endl;
        }

        {
            auto start = std::chrono::high_resolution_clock::now();
            for (const double& E : test_points) {
                volatile double result = interpolate_binary_search_cpp(
                    T_arr, E, E_arr);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count();
            std::cout << "Binary Search (array) time: " << duration << " ns" << std::endl;
        }

        // Measure double lookup performance
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (const double& E : test_points) {
                volatile double result = interpolate_double_lookup_cpp(
                    E, T_vec, E_vec, E_eq_vec, inv_delta_E_eq, idx_map_vec);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count();
            std::cout << "Double Lookup (vector) time: " << duration << " ns" << std::endl;
        }

        {
            auto start = std::chrono::high_resolution_clock::now();
            for (const double& E : test_points) {
                volatile double result = interpolate_double_lookup_cpp(
                    E, T_arr, E_arr, E_eq_arr, inv_delta_E_eq, idx_map_arr);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count();
            std::cout << "Double Lookup (array) time: " << duration << " ns" << std::endl;
        }
    }
}


int main() {
    try {
        test_basic_functionality();
        test_edge_cases_and_errors();
        test_interpolation_accuracy();
        test_performance();

        std::cout << "\nAll tests completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
