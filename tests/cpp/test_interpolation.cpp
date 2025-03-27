#include "pymatlib_interpolators/interpolate_binary_search_cpp.h"
#include "pymatlib_interpolators/interpolate_double_lookup_cpp.h"
#include <array>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>
#include <iostream>
#include <numeric>
#include "TestArrayContainer.hpp"

// Helper function to compare floating point numbers
bool is_equal(const double a, const double b, double tolerance = 1e-10) {
    return std::abs(a - b) < tolerance;
}

void test_basic_functionality() {
    std::cout << "\nTesting basic functionality..." << std::endl;

    // Test middle point interpolation
    constexpr double test_E = 16950000000.0;

    // Binary Search Tests
    {
        constexpr BinarySearchTests tests;
        const double result = tests.interpolateBS(test_E);
        // Expected value based on T_bs array
        constexpr double expected_T = 3253.15;
        assert(is_equal(result, expected_T));
        std::cout << "Basic binary search interpolation passed" << std::endl;
    }

    // Double Lookup Tests
    {
        constexpr DoubleLookupTests tests;
        const double result = tests.interpolateDL(test_E);
        // Expected value based on T_eq array
        constexpr double expected_T = 3258.15;
        assert(is_equal(result, expected_T));
        std::cout << "Basic double lookup interpolation passed" << std::endl;
    }
}

void test_edge_cases_and_errors() {
    std::cout << "\nTesting edge cases and error handling..." << std::endl;

    // Test boundary values for binary search
    {
        BinarySearchTests tests;

        // Test below minimum
        const double result_min = tests.interpolateBS(16750000000.0);
        assert(is_equal(result_min, 3243.15));

        // Test above maximum
        const double result_max = tests.interpolateBS(17150000000.0);
        assert(is_equal(result_max, 3278.15));

        std::cout << "Edge cases for binary search passed" << std::endl;
    }

    // Double Lookup edge cases
    {
        DoubleLookupTests tests;

        // Test below minimum
        const double result_min = tests.interpolateDL(16750000000.0);
        assert(is_equal(result_min, 3243.15));

        // Test above maximum
        const double result_max = tests.interpolateDL(17150000000.0);
        assert(is_equal(result_max, 3273.15));

        std::cout << "Edge cases for double lookup passed" << std::endl;
    }
}

void test_interpolation_accuracy() {
    std::cout << "\nTesting interpolation accuracy..." << std::endl;

    // Test data with precise values
    struct TestCase {
        double input_E;
        double expected_T_bs;
        double expected_T_dl;
        double tolerance;
        std::string description;
    };

    const std::vector<TestCase> test_cases = {
        // Test cases with expected interpolated values for both methods
        {16850000000.0, 3245.65, 3248.15, 1e-10, "Quarter-point interpolation"},
        {16950000000.0, 3253.15, 3258.15, 1e-10, "Mid-point interpolation"},
        {17050000000.0, 3268.15, 3268.15, 1e-10, "Three-quarter-point interpolation"},
        // Near exact points
        {16900000000.0, 3248.15, 3253.15, 1e-8, "Near-exact point"},
        {17000000000.0, 3258.15, 3263.15, 1e-8, "Near-exact point"}
    };

    for (const auto& test : test_cases) {
        // Binary Search Tests
        BinarySearchTests bs_tests;
        const double result_bs = bs_tests.interpolateBS(test.input_E);
        assert(is_equal(result_bs, test.expected_T_bs, test.tolerance));

        // Double Lookup Tests
        DoubleLookupTests dl_tests;
        const double result_dl = dl_tests.interpolateDL(test.input_E);
        assert(is_equal(result_dl, test.expected_T_dl, test.tolerance));

        std::cout << "Accuracy test passed for " << test.description << std::endl;
    }
}

void test_consistency() {
    std::cout << "\nTesting interpolation consistency..." << std::endl;

    // Test that small changes in input produce correspondingly small changes in output
    constexpr BinarySearchTests bs_tests;
    constexpr DoubleLookupTests dl_tests;

    constexpr double base_E = 16900000000.0;
    constexpr double delta_E = 1000000.0;  // Small change in energy

    const double base_T_bs = bs_tests.interpolateBS(base_E);
    const double delta_T_bs = bs_tests.interpolateBS(base_E + delta_E) - base_T_bs;

    const double base_T_dl = dl_tests.interpolateDL(base_E);
    const double delta_T_dl = dl_tests.interpolateDL(base_E + delta_E) - base_T_dl;

    // Check that changes are reasonable (not too large for small input change)
    assert(std::abs(delta_T_bs) < 1.0);
    assert(std::abs(delta_T_dl) < 1.0);

    std::cout << "Consistency test passed" << std::endl;
}

void test_stress() {
    std::cout << "\nPerforming stress testing..." << std::endl;

    constexpr BinarySearchTests bs_tests;
    constexpr DoubleLookupTests dl_tests;

    // Test with extremely large values
    constexpr double large_E = 1.0e20;
    const double result_large_bs = bs_tests.interpolateBS(large_E);
    const double result_large_dl = dl_tests.interpolateDL(large_E);

    // Test with extremely small values
    constexpr double small_E = 1.0e-20;
    const double result_small_bs = bs_tests.interpolateBS(small_E);
    const double result_small_dl = dl_tests.interpolateDL(small_E);

    // For extreme values, we should get boundary values
    assert(is_equal(result_large_bs, 3278.15));
    assert(is_equal(result_large_dl, 3273.15));
    assert(is_equal(result_small_bs, 3243.15));
    assert(is_equal(result_small_dl, 3243.15));

    std::cout << "Stress tests passed" << std::endl;
}

void test_random_validation() {
    std::cout << "\nTesting with random inputs..." << std::endl;

    constexpr BinarySearchTests bs_tests;
    constexpr DoubleLookupTests dl_tests;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(16800000000.0, 17100000000.0);

    constexpr int num_tests = 1000;
    for (int i = 0; i < num_tests; ++i) {
        const double random_E = dist(gen);

        const double result_bs = bs_tests.interpolateBS(random_E);
        const double result_dl = dl_tests.interpolateDL(random_E);

        // Results should be within the temperature range
        assert(result_bs >= 3243.15 && result_bs <= 3278.15);
        assert(result_dl >= 3243.15 && result_dl <= 3273.15);
    }

    std::cout << "Random input validation passed" << std::endl;
}


void test_performance() {
    std::cout << "\nTesting performance..." << std::endl;

    // Configuration parameters
    constexpr int warmupSteps = 5;
    constexpr int outerIterations = 10;
    constexpr int numCells = 64*64*64;

    // Setup test data
    constexpr SS304L test;
    std::vector<double> random_energies(numCells);

    // Generate random values
    std::random_device rd;
    std::mt19937 gen(rd());
    constexpr double y_min = SS304L::y_neq.front() * 0.8;
    constexpr double y_max = SS304L::y_neq.back() * 1.2;
    std::uniform_real_distribution<double> dist(y_min, y_max);

    // Fill random energies
    for(auto& E : random_energies) {
        E = dist(gen);
    }

    // Warmup runs
    std::cout << "Performing warmup steps..." << std::endl;
    for(int i = 0; i < warmupSteps; ++i) {
        for(const double& E : random_energies) {
            // volatile double result = interpolate_double_lookup_cpp(E, test);
            volatile double result = test.interpolateDL(E);
        }
    }
    for(int i = 0; i < warmupSteps; ++i) {
        for(const double& E : random_energies) {
            volatile double result = test.interpolateBS(E);
        }
    }

    // Performance measurement
    std::cout << "\nStarting performance measurement..." << std::endl;
    std::vector<double> timings_binary;
    std::vector<double> timings_double_lookup;

    for(int iter = 0; iter < outerIterations; ++iter) {
        std::cout << "\nIteration " << iter + 1 << "/" << outerIterations << std::endl;

        // Double Lookup timing
        {
            const auto start1 = std::chrono::high_resolution_clock::now();
            for(const double& E : random_energies) {
                volatile double result = test.interpolateDL(E);
            }
            const auto end1 = std::chrono::high_resolution_clock::now();
            const auto duration1 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end1 - start1).count();
            timings_double_lookup.push_back(static_cast<double>(duration1));

            std::cout << "Double Lookup - Iteration time: " << duration1 << " ns" << std::endl;
        }

        // Binary Search timing
        {
            const auto start2 = std::chrono::high_resolution_clock::now();
            for(const double& E : random_energies) {
                volatile double result = test.interpolateBS(E);
            }
            const auto end2 = std::chrono::high_resolution_clock::now();
            const auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(
                end2 - start2).count();
            timings_binary.push_back(static_cast<double>(duration2));

            std::cout << "Binary Search - Iteration time: " << duration2 << " ns" << std::endl;
        }
    }

    // Calculate and print statistics
    auto calc_stats = [](const std::vector<double>& timings) {
        const double sum = std::accumulate(timings.begin(), timings.end(), 0.0);
        double mean = sum / static_cast<double>(timings.size());
        const double sq_sum = std::inner_product(timings.begin(), timings.end(),timings.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / static_cast<double>(timings.size()) - mean * mean);
        return std::make_pair(mean, stdev);
    };

    auto [binary_mean, binary_stdev] = calc_stats(timings_binary);
    auto [lookup_mean, lookup_stdev] = calc_stats(timings_double_lookup);

    std::cout << "\nPerformance Results (" << numCells << " cells, "
              << outerIterations << " iterations):" << std::endl;
    std::cout << "Binary Search:" << std::endl;
    std::cout << "  Mean time: " << binary_mean << " ± " << binary_stdev << " ns" << std::endl;
    std::cout << "  Per cell: " << binary_mean/numCells << " ns" << std::endl;

    std::cout << "Double Lookup:" << std::endl;
    std::cout << "  Mean time: " << lookup_mean << " ± " << lookup_stdev << " ns" << std::endl;
    std::cout << "  Per cell: " << lookup_mean/numCells << " ns" << std::endl;
}


int main() {
    try {
        test_basic_functionality();
        test_edge_cases_and_errors();
        test_interpolation_accuracy();
        test_consistency();
        test_stress();
        test_random_validation();
        test_performance();

        std::cout << "\nAll tests completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
