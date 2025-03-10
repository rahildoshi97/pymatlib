# Interpolation Methods in pymatlib

This document explains the interpolation techniques used in pymatlib for energy-temperature conversion,
including their internal workings, automatic method selection, and performance considerations.

### Overview

Interpolation is essential in pymatlib for converting between energy density and temperature values, particularly when modeling materials whose properties vary significantly with temperature. pymatlib provides two interpolation methods:

- Binary Search Interpolation
- Double Lookup Interpolation

Additionally, pymatlib offers an intelligent wrapper method that automatically selects the optimal interpolation method based on data characteristics.

## Binary Search Interpolation (`interpolateBS`)

Binary Search Interpolation is one of the primary methods used in pymatlib for obtaining temperatures from energy density values.

### How It Works

1. The algorithm starts with a sorted array of energy density values and their corresponding temperature values
2. When given an energy density value to convert to temperature:
- It performs a binary search to find the position of the target value
- It divides the search interval in half repeatedly until finding the closest match
- Once the position is found, it performs linear interpolation between the two closest points

### Implementation Details

The binary search implementation handles edge cases gracefully:
- If the target energy is below the minimum value, it returns the lowest temperature
- If the target energy is above the maximum value, it returns the highest temperature
- For values within the range, it calculates the interpolated temperature using:
```text
T = T[i] + (T[i+1] - T[i]) * (E_target - E[i]) / (E[i+1] - E[i])
```

As shown in the test code, this method is robust and handles a wide range of inputs, including extreme values and edge cases.

### Characteristics
- **Time Complexity**: O(log n) where n is the number of data points
- **Advantages**: Works reliably with any monotonically increasing data
- **Implementation**: Available through the `interpolateBS()` method in generated C++ code

## Double Lookup Interpolation (`interpolateDL`)

Double Lookup Interpolation is an optimized approach designed specifically for uniform or near-uniform data distributions.

### How It Works

1. During initialization (Python)
- The algorithm checks if the temperature array is equidistant using `check_equidistant()`
- It creates a uniform grid of energy values (`E_eq`) with carefully calculated spacing:
```python
delta_min = np.min(np.abs(np.diff(E_neq)))
delta_E_eq = max(np.floor(delta_min * 0.95), 1.)
E_eq = np.arange(E_neq[0], E_neq[-1] + delta_E_eq, delta_E_eq)
```
- It pre-computes an index mapping between the original energy array and the uniform grid:
```python
idx_map = np.searchsorted(E_neq, E_eq, side='right') - 1
idx_map = np.clip(idx_map, 0, len(E_neq) - 2)
```
- It calculates the inverse of the energy grid spacing for fast index calculation:

```python
inv_delta_E_eq = 1.0 / (E_eq[1] - E_eq[0])
```

2. During lookup (C++):
- It calculates the array index directly (O(1)) using the pre-computed inverse delta:
```python
idx = floor((E_target - E_eq[0]) * inv_delta_E_eq)
```
- It retrieves the pre-computed mapping to find the correct segment in the original array
- It performs a simple linear interpolation between the two closest points

### Characteristics
- **Time Complexity**: O(1) - constant time regardless of array size using pre-computed mappings
- **Best For**: Uniform temperature distributions
- **Advantages**: Significantly faster (typically 2-4x) than binary search for uniform data
- **Implementation**: Available through the `interpolateDL()` method in generated C++ code

### Performance Optimization

The double lookup method achieves O(1) complexity through several optimizations:
- Pre-computing the index mapping eliminates the need for binary search
- Using the inverse of the delta for multiplication instead of division
- Constraining the index mapping to valid bounds during initialization

## Automatic Method Selection Process (`interpolate`)

```c++
constexpr double energy_density = 16950000000.0;
constexpr SimpleSteel alloy;
const double temperature = alloy.interpolate(energy_density);
```

How Automatic Selection Works Internally

The automatic method selection in pymatlib involves two key components working together:

1. **Analysis Phase (Python)**: The prepare_interpolation_arrays() function analyzes the input temperature and energy density arrays to determine the most appropriate interpolation method:
- It validates that both arrays are monotonic and energy increases with temperature
- It checks if the temperature array is equidistant using check_equidistant()
- Based on this analysis, it selects either "binary_search" or "double_lookup" as the method
- It returns a dictionary containing the processed arrays and the selected method

```python
if is_equidistant:
    E_eq, inv_delta_E_eq = E_eq_from_E_neq(E_bs)
    idx_mapping = create_idx_mapping(E_bs, E_eq)
    # Set method to "double_lookup"
else:
    # Set method to "binary_search"
```

2. **Code Generation Phase (C++)**: The InterpolationArrayContainer class uses this information to generate optimized C++ code:
- It always includes the binary search method (interpolateBS)
- If the temperature array is equidistant, it also includes the double lookup method (interpolateDL)
- It adds an intelligent wrapper method interpolate() that automatically calls the preferred method:

```c++
// If double lookup is available, use it
if (self.has_double_lookup):
    interpolate() { return interpolate_double_lookup_cpp(E_target, *this); }
// Otherwise, fall back to binary search
else:
    interpolate() { return interpolate_binary_search_cpp(E_target, *this); }
```

This two-step process ensures that:

- The most efficient method is selected based on the data characteristics
- Users can simply call `material.interpolate(energy_density)` without worrying about which technique to use

This method:
- Uses Double Lookup if the data is suitable (uniform or near-uniform)
- Falls back to Binary Search for non-uniform data
- Provides the best performance for your specific data structure without manual selection
- Performance is optimized automatically without manual intervention

## Performance Considerations

Based on performance testing with 64³ cells:

- **Binary Search**: Reliable but slower for large datasets
- **Double Lookup**: Typically 2-4x faster than Binary Search for uniform data
- **Edge Cases**: Both methods handle values outside the data range by clamping to the nearest endpoint

The test results demonstrate that Double Lookup consistently outperforms Binary Search when the data is suitable, 
making it the preferred method for uniform temperature distributions.

## Implementation Details

The interpolation methods are implemented in C++ for maximum performance:

- The `InterpolationArrayContainer` class generates optimized C++ code with the interpolation arrays and methods
- The generated code includes both interpolation methods and the automatic selector
- The implementation handles edge cases gracefully, returning boundary values for out-of-range queries

## Edge Case Handling

Both interpolation methods include robust handling of edge cases:

- **Out-of-Range Values**: Both methods clamp values outside the defined range to the nearest endpoint

- **Extreme Values**: As shown in the stress tests, both methods handle extremely large and small values correctly

- **Monotonicity**: The code validates that both arrays are monotonic and that energy increases with temperature

The test file demonstrates this with specific test cases:

```c++
// Test with extremely large values
constexpr double large_E = 1.0e20;
const double result_large_bs = bs_tests.interpolateBS(large_E);
const double result_large_dl = dl_tests.interpolateDL(large_E);

// Test with extremely small values
constexpr double small_E = 1.0e-20;
const double result_small_bs = bs_tests.interpolateBS(small_E);
const double result_small_dl = dl_tests.interpolateDL(small_E);
```

## When to Use Each Method

- **Binary Search**: Use for general-purpose interpolation or when data points are irregularly spaced
- **Double Lookup**: Use when data points are uniformly or nearly uniformly spaced
- **Automatic Selection**: Use in most cases to get the best performance without manual selection

For most applications, the automatic `interpolate()` method provides the best balance of performance and reliability.
