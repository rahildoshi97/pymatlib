#include "interpolate_binary_search_cpp.h"
#include "interpolate_double_lookup_cpp.h"
#include <array>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <iostream>
#include "TestArrayContainer.hpp"

/*
void run_comprehensive_tests1() {
    std::cout << "\nRunning comprehensive interpolation tests...\n" << std::endl;

    // Test Case 1: Basic ascending arrays
    {
        std::vector<double> T = {100.0, 200.0, 300.0, 400.0, 500.0};
        std::vector<double> E = {1.0, 2.0, 3.0, 4.0, 5.0};

        // Test middle interpolation
        double result = interpolate_binary_search_cpp(T, 2.5, E);
        assert(std::abs(result - 250.0) < 1e-8);
        std::cout << "Basic ascending test passed" << std::endl;
    }

    // Test Case 2: Descending arrays
    {
        std::vector<double> T = {500.0, 400.0, 300.0, 200.0, 100.0};
        std::vector<double> E = {5.0, 4.0, 3.0, 2.0, 1.0};

        double result = interpolate_binary_search_cpp(T, 2.5, E);
        assert(std::abs(result - 250.0) < 1e-8);
        std::cout << "Descending array test passed" << std::endl;
    }

    // Test Case 3: Edge cases
    {
        std::vector<double> T = {100.0, 200.0, 300.0};
        std::vector<double> E = {1.0, 2.0, 3.0};

        // Test below minimum
        double result = interpolate_binary_search_cpp(T, 0.5, E);
        assert(std::abs(result - 100.0) < 1e-8);

        // Test above maximum
        result = interpolate_binary_search_cpp(T, 3.5, E);
        assert(std::abs(result - 300.0) < 1e-8);

        std::cout << "Edge cases test passed" << std::endl;
    }

    // Test Case 4: Double lookup specific tests
    {
        std::vector<double> T_eq = {100.0, 200.0, 300.0, 400.0, 500.0};
        std::vector<double> E_neq = {1.0, 2.1, 3.4, 4.2, 5.0};
        std::vector<double> E_eq = {1.0, 2.0, 3.0, 4.0, 5.0};
        double inv_delta_E_eq = 1.0;
        std::vector<double> idx_map = {0, 0, 1, 2, 3};

        double result = interpolate_double_lookup_cpp(2.5, T_eq, E_neq, E_eq,
                                                    inv_delta_E_eq, idx_map);
        std::cout << result << std::endl;
        // assert(std::abs(result - 230.769) < 1e-6);
        std::cout << "Double lookup basic test passed" << std::endl;
    }

    // Test Case 5: Performance comparison
    {
        const size_t size = 1000000;
        std::vector<double> T(size);
        std::vector<double> E(size);
        std::vector<double> E_eq(size);
        std::vector<double> idx_map(size);

        // Initialize arrays
        for(size_t i = 0; i < size; ++i) {
            T[i] = static_cast<double>(i);
            E[i] = static_cast<double>(i) * 1.5;
            E_eq[i] = static_cast<double>(i);
            idx_map[i] = i;
        }

        double inv_delta_E_eq = 1.0;

        // Time binary search
        auto start = std::chrono::high_resolution_clock::now();
        double result1 = interpolate_binary_search_cpp(T, 500000.5, E);
        std::cout << result1 << std::endl;
        auto end = std::chrono::high_resolution_clock::now();
        auto binary_duration = std::chrono::duration_cast<std::chrono::nanoseconds>
                             (end - start).count();

        // Time double lookup
        start = std::chrono::high_resolution_clock::now();
        double result2 = interpolate_double_lookup_cpp(500000.5, T, E, E_eq,
                                                     inv_delta_E_eq, idx_map);
        std::cout << result2 << std::endl;
        end = std::chrono::high_resolution_clock::now();
        auto double_duration = std::chrono::duration_cast<std::chrono::nanoseconds>
                             (end - start).count();

        std::cout << "\nPerformance test results:" << std::endl;
        std::cout << "Binary search: " << binary_duration << " ns" << std::endl;
        std::cout << "Double lookup: " << double_duration << " ns" << std::endl;
    }

    // Test Case 6: Error handling
    {
        try {
            std::vector<double> T = {100.0};
            std::vector<double> E = {1.0};
            interpolate_binary_search_cpp(T, 1.5, E);
            assert(false && "Should throw exception for size < 2");
        } catch (const std::runtime_error&) {
            std::cout << "Size validation test passed" << std::endl;
        }
    }

    // Test Case 7: Non-uniform spacing
    {
        std::vector<double> T = {100.0, 150.0, 300.0, 320.0, 500.0};
        std::vector<double> E = {1.0, 1.5, 3.0, 3.2, 5.0};

        double result = interpolate_binary_search_cpp(T, 2.0, E);
        std::cout << "Non-uniform spacing test passed" << std::endl;
    }
}
*/

/*
void run_comprehensive_tests2() {
    std::cout << "\nRunning comprehensive interpolation tests...\n" << std::endl;

    // Test Case 1: Basic tests with both std::vector and std::array
    {
        // Vector version
        std::vector<double> T_vec = {100.0, 200.0, 300.0, 400.0};
        std::vector<double> E_vec = {1.0, 2.0, 3.0, 4.0};
        double result_vec = interpolate_binary_search_cpp(T_vec, 2.5, E_vec);
        assert(std::abs(result_vec - 250.0) < 1e-6);

        // Array version
        std::array<double, 4> T_arr = {100.0, 200.0, 300.0, 400.0};
        std::array<double, 4> E_arr = {1.0, 2.0, 3.0, 4.0};
        double result_arr = interpolate_binary_search_cpp(T_arr, 2.5, E_arr);
        assert(std::abs(result_arr - 250.0) < 1e-6);

        std::cout << "Basic vector/array tests passed" << std::endl;
    }

    // Test Case 2: Descending order with both containers
    {
        // Vector version
        std::vector<double> T_vec = {400.0, 300.0, 200.0, 100.0};
        std::vector<double> E_vec = {4.0, 3.0, 2.0, 1.0};
        double result_vec = interpolate_binary_search_cpp(T_vec, 2.5, E_vec);

        // Array version
        std::array<double, 4> T_arr = {400.0, 300.0, 200.0, 100.0};
        std::array<double, 4> E_arr = {4.0, 3.0, 2.0, 1.0};
        double result_arr = interpolate_binary_search_cpp(T_arr, 2.5, E_arr);

        assert(std::abs(result_vec - result_arr) < 1e-8);
        std::cout << "Descending order vector/array tests passed" << std::endl;
    }

    // Test Case 3: Edge cases for both containers
    {
        // Vector version
        std::vector<double> T_vec = {3273.15, 3263.15, 3253.15, 3243.15};
        std::vector<double> E_vec = {1.71e10, 1.70e10, 1.69e10, 1.68e10};

        double result_vec_min = interpolate_binary_search_cpp(T_vec, 1.67e10, E_vec);
        double result_vec_max = interpolate_binary_search_cpp(T_vec, 1.72e10, E_vec);

        // Array version
        std::array<double, 4> T_arr = {3273.15, 3263.15, 3253.15, 3243.15};
        std::array<double, 4> E_arr = {1.71e10, 1.70e10, 1.69e10, 1.68e10};

        double result_arr_min = interpolate_binary_search_cpp(T_arr, 1.67e10, E_arr);
        double result_arr_max = interpolate_binary_search_cpp(T_arr, 1.72e10, E_arr);

        assert(std::abs(result_vec_min - 3243.15) < 1e-6);
        assert(std::abs(result_arr_min - 3243.15) < 1e-6);
        assert(std::abs(result_vec_max - 3273.15) < 1e-6);
        assert(std::abs(result_arr_max - 3273.15) < 1e-6);

        std::cout << "Edge cases vector/array tests passed" << std::endl;
    }

    // Test Case 4: Double lookup with both containers
    {
        // Vector version
        std::vector<double> T_vec = {100.0, 200.0, 300.0, 400.0};
        std::vector<double> E_neq_vec = {1.0, 2.1, 3.4, 4.2};
        std::vector<double> E_eq_vec = {1.0, 2.0, 3.0, 4.0};
        std::vector<double> idx_map_vec = {0, 0, 1, 2};

        // Array version
        std::array<double, 4> T_arr = {100.0, 200.0, 300.0, 400.0};
        std::array<double, 4> E_neq_arr = {1.0, 2.1, 3.4, 4.2};
        std::array<double, 4> E_eq_arr = {1.0, 2.0, 3.0, 4.0};
        std::array<double, 4> idx_map_arr = {0, 0, 1, 2};

        double inv_delta_E_eq = 1.0;

        double result_vec = interpolate_double_lookup_cpp(2.5, T_vec, E_neq_vec,
                                                        E_eq_vec, inv_delta_E_eq,
                                                        idx_map_vec);

        double result_arr = interpolate_double_lookup_cpp(2.5, T_arr, E_neq_arr,
                                                        E_eq_arr, inv_delta_E_eq,
                                                        idx_map_arr);

        assert(std::abs(result_vec - result_arr) < 1e-6);
        std::cout << "Double lookup vector/array tests passed" << std::endl;
    }

    // Test Case 5: Error handling for both containers
    {
        try {
            std::vector<double> T_vec = {100.0};
            std::vector<double> E_vec = {1.0};
            interpolate_binary_search_cpp(T_vec, 1.5, E_vec);
            assert(false && "Should throw exception for vector size < 2");
        } catch (const std::runtime_error&) {
            std::cout << "Vector size validation passed" << std::endl;
        }

        try {
            std::array<double, 1> T_arr = {100.0};
            std::array<double, 1> E_arr = {1.0};
            interpolate_binary_search_cpp(T_arr, 1.5, E_arr);
            assert(false && "Should throw exception for array size < 2");
        } catch (const std::runtime_error&) {
            std::cout << "Array size validation passed" << std::endl;
        }
    }

    // Test Case 6: Performance comparison between vector and array
    {
        const size_t size = 1000000;
        std::vector<double> T_vec(size);
        std::vector<double> E_vec(size);
        std::array<double, 1000> T_arr;
        std::array<double, 1000> E_arr;

        // Initialize containers
        for(size_t i = 0; i < size; ++i) {
            T_vec[i] = static_cast<double>(i);
            E_vec[i] = static_cast<double>(i) * 1.5;
        }
        for(size_t i = 0; i < 1000; ++i) {
            T_arr[i] = static_cast<double>(i);
            E_arr[i] = static_cast<double>(i) * 1.5;
        }

        // Time vector version
        auto start = std::chrono::high_resolution_clock::now();
        double result_vec = interpolate_binary_search_cpp(T_vec, 500000.5, E_vec);
        auto vector_duration = std::chrono::duration_cast<std::chrono::nanoseconds>
                             (std::chrono::high_resolution_clock::now() - start).count();

        // Time array version
        start = std::chrono::high_resolution_clock::now();
        double result_arr = interpolate_binary_search_cpp(T_arr, 500.5, E_arr);
        auto array_duration = std::chrono::duration_cast<std::chrono::nanoseconds>
                            (std::chrono::high_resolution_clock::now() - start).count();

        std::cout << "\nPerformance comparison:" << std::endl;
        std::cout << "Vector duration: " << vector_duration << " ns" << std::endl;
        std::cout << "Array duration:  " << array_duration << " ns" << std::endl;
    }
}
*/


// Helper function to compare floating point numbers
bool is_equal(double a, double b, double tolerance = 1e-10) {
    return std::abs(a - b) < tolerance;
}


void test_basic_functionality1() {
    // Use realistic temperature-enthalpy values from search results
    const std::vector<double> T_vec = {2400.0, 2500.0, 2600.0, 2700.0};
    const std::vector<double> E_vec = {1.68e10, 1.69e10, 1.70e10, 1.71e10};

    // Test point should be within the data range
    const double test_E = 1.695e10;
    const double expected_T = 2550.0;  // Midpoint interpolation

    // Add tolerance appropriate for temperature values
    const double tolerance = 1e-6;

    DoubleLookupTests tests;
    double result_vec = interpolate_binary_search_cpp(test_E, tests);
    assert(std::abs(result_vec - expected_T) < tolerance);
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
        DoubleLookupTests tests;
        double result_arr = interpolate_binary_search_cpp(test_E, tests);
        assert(is_equal(result_arr, expected_T));

        // Verify vector and array results match
        assert(is_equal(result_vec, result_arr));
    }

    // Double Lookup Tests
    {
        // Setup double lookup specific data
        // std::vector<double> E_eq_vec = {1.68e10, 1.69e10, 1.70e10, 1.71e10};
        std::array<double, 4> E_eq_arr = {1.68e10, 1.69e10, 1.70e10, 1.71e10};
        // std::vector<double> idx_map_vec = {0, 1, 2, 3};
        std::array<double, 4> idx_map_arr = {0, 1, 2, 3};
        const double inv_delta_E_eq = 1.0 / (1e9);

        /*double result_vec = interpolate_double_lookup_cpp(
            test_E, T_vec, E_vec, E_eq_vec, inv_delta_E_eq, idx_map_vec);
        assert(is_equal(result_vec, expected_T));*/

        DoubleLookupTests tests;
        double result_arr = tests.interpolateDL(test_E);
        /*double result_arr = interpolate_double_lookup_cpp(
            test_E, T_arr, E_arr, E_eq_arr, inv_delta_E_eq, idx_map_arr);*/
        assert(is_equal(result_arr, expected_T));

        // Verify vector and array results match
        // assert(is_equal(result_vec, result_arr));
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
        /*double result_vec_min = interpolate_binary_search_cpp(T_vec, 1.67e10, E_vec);
        double result_vec_max = interpolate_binary_search_cpp(T_vec, 1.72e10, E_vec);
        assert(is_equal(result_vec_min, 3243.15));
        assert(is_equal(result_vec_max, 3273.15));*/

        // Array tests
        DoubleLookupTests tests;
        double result_arr_min = interpolate_binary_search_cpp(1.67e10, tests);
        double result_arr_max = interpolate_binary_search_cpp(1.72e10, tests);
        assert(is_equal(result_arr_min, 3243.15));
        assert(is_equal(result_arr_max, 3273.15));
    }

    // Error cases
    /*{
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
    }*/

    // Double Lookup edge cases
    {
        std::vector<double> E_eq_vec = {1.68e10, 1.69e10, 1.70e10, 1.71e10};
        std::vector<double> idx_map_vec = {0, 1, 2, 3};
        std::array<double, 4> E_eq_arr = {1.68e10, 1.69e10, 1.70e10, 1.71e10};
        std::array<double, 4> idx_map_arr = {0, 1, 2, 3};
        const double inv_delta_E_eq = 1.0 / (1e9);



        // Array tests
        DoubleLookupTests tests;
        double result_arr_min = tests.interpolateDL(1.67e10);
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

    DoubleLookupTests tests;
    for (const auto& test : test_cases) {
        // Binary Search Tests
        double result_bin_arr = interpolate_binary_search_cpp(test.input_E, tests);

        assert(is_equal(result_bin_vec, test.expected_T, test.tolerance));
        assert(is_equal(result_bin_arr, test.expected_T, test.tolerance));
        assert(is_equal(result_bin_vec, result_bin_arr, test.tolerance));

        // Double Lookup Tests
        double result_dl_arr = tests.interpolateDL(test.input_E);

        // assert(is_equal(result_dl_vec, test.expected_T, test.tolerance));
        assert(is_equal(result_dl_arr, test.expected_T, test.tolerance));
        // assert(is_equal(result_dl_vec, result_dl_arr, test.tolerance));
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
            DoubleLookupTests tests;
            auto start = std::chrono::high_resolution_clock::now();
            for (const double& E : test_points) {
                volatile double result = interpolate_binary_search_cpp(E, tests);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count();
            std::cout << "Binary Search (vector) time: " << duration << " ns" << std::endl;
        }

        {
            DoubleLookupTests tests;
            auto start = std::chrono::high_resolution_clock::now();
            for (const double& E : test_points) {
                volatile double result = interpolate_binary_search_cpp(E, tests);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count();
            std::cout << "Binary Search (array) time: " << duration << " ns" << std::endl;
        }

        // Measure double lookup performance
        {
            DoubleLookupTests tests;
            auto start = std::chrono::high_resolution_clock::now();
            for (const double& E : test_points) {
                volatile double result = tests.interpolateDL(E);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end - start).count();
            std::cout << "Double Lookup (vector) time: " << duration << " ns" << std::endl;
        }

        {
            auto start = std::chrono::high_resolution_clock::now();
            for (const double& E : test_points) {
                DoubleLookupTests tests;
                volatile double result = tests.interpolateDL(E);
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
