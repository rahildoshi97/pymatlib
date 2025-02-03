#include "interpolate_binary_search_cpp.h"
#include <vector>
#include <array>
#include <iostream>
#include <cassert>

void run_cpp_tests() {
    // Test with std::vector
    std::vector<double> temperatures = {3273.15, 3263.15, 3253.15, 3243.15};
    std::vector<double> energy_densities = {1.71e10, 1.70e10, 1.69e10, 1.68e10};


    try {
        // Test normal interpolation
        double h_in = 1.695e10;
        double result = interpolate_binary_search_cpp(
            temperatures, h_in, energy_densities);
        std::cout << "Vector test temperature: " << result << std::endl;
        assert(result > 3243.15 && result < 3273.15);

        // Test with std::array
        std::array<double, 4> temp_arr = {3273.15, 3263.15, 3253.15, 3243.15};
        std::array<double, 4> energy_arr = {1.71e10, 1.70e10, 1.69e10, 1.68e10};
        result = interpolate_binary_search_cpp(temp_arr, h_in, energy_arr);
        std::cout << "Array test temperature: " << result << std::endl;
        assert(result > 3243.15 && result < 3273.15);

        // Test boundary values
        result = interpolate_binary_search_cpp(
            temperatures, 1.67e10, energy_densities);
        assert(std::abs(result - 3243.15) < 1e-6);

        result = interpolate_binary_search_cpp(
            temperatures, 1.72e10, energy_densities);
        assert(std::abs(result - 3273.15) < 1e-6);

        // Test error cases
        std::vector<double> wrong_size = {1.0, 2.0};
        try {
            interpolate_binary_search_cpp(temperatures, h_in, wrong_size);
            std::cerr << "Expected size mismatch error" << std::endl;
            assert(false);
        } catch (const std::runtime_error&) {
            std::cout << "Size mismatch test passed" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Test error: " << e.what() << std::endl;
        assert(false);
    }
}

int main() {
    run_cpp_tests();
    std::cout << "All C++ tests completed successfully" << std::endl;
    return 0;
}
