# Defining Custom Material Properties

This guide explains how to define custom material properties in PyMatLib using different methods.

## YAML Configuration Options

PyMatLib supports several ways to define material properties in YAML files, following SI units (m, s, kg, A, V, K, etc.).

### 1. Constant Value

For properties that don't vary with temperature:

```yaml
properties:
    thermal_expansion_coefficient: 16.3e-6
```

### 2. Step Functions

For properties that change abruptly at phase transitions:
```YAML
properties:
    latent_heat_of_fusion:
    temperature: solidus_temperature
    value: [0.0, 171401.0]
    bounds: [constant, constant]
```

### 3. Importing from External Files

For properties defined in spreadsheets:

```yaml
properties:
    # Excel file format
    density:
        file_path: ./304L_data.xlsx
        temperature_column: T (K)
        property_column: Density (kg/(m)^3)
        bounds: [constant, constant]
    
    # CSV file format
    heat_capacity:
      file_path: ./heat_capacity_data.csv
      temperature_column: Temperature
      property_column: Cp
      bounds: [constant, constant]
    
    # Text file format (space/tab separated)
    thermal_conductivity:
      file_path: ./conductivity_data.txt
      temperature_column: 0  # Column index for headerless files
      property_column: 1
      bounds: [constant, constant]
```
Supported file formats include .txt (space/tab separated), .csv, and .xlsx.

### 4. Tabular Data for Interpolation

For properties that vary with temperature:

```yaml
properties:
    # Using explicit temperature list
    heat_conductivity:
        temperature: [1200, 1800, 2200, 2400]  # Temperatures in Kelvin
        value: [25, 30, 33, 35]  # Property values
        bounds: [constant, constant]
    
    # Using references to defined temperatures
    latent_heat_of_fusion:
      temperature: [solidus_temperature, liquidus_temperature]
      value: [171401]
      bounds: [constant, constant]
    
    # Using tuple for temperature generation
    heat_capacity:
      temperature: (1000, 200)  # Start at 1000K and increment by 200K for each value
      value:
      bounds: [constant, constant]
    
    # Using tuple with negative increment
    density:
      temperature: (1735.00, -5)  # Start at 1735.00K and decrement by 5K for each value
      value: [7037.470, 7060.150, 7088.800, 7110.460, 7127.680, 7141.620, 7156.800, 7172.590, 7184.010, 7192.780]
      bounds: [constant, constant]
```

### 5. Piecewise Equations

For properties with different equations in different temperature ranges:
```yaml
properties:
    heat_conductivity:
    temperature: [1700][3000]
    equation: ["0.012T + 13", "0.015T + 5"]
    bounds: [constant, constant]
```

### 6. Computed Properties

For properties that can be derived from others:

```yaml
properties:
    # Thermal diffusivity computation
    thermal_diffusivity:
    temperature: (300, 3000, 5.0)
    equation: heat_conductivity / (density * heat_capacity)
    bounds: [extrapolate, extrapolate]
    
    # Energy density with different models
    energy_density:
      temperature: (300, 3000, 541)  # 541 evenly spaced points
      equation: density * specific_enthalpy
      bounds: [extrapolate, extrapolate]
```

## Temperature Definition Formats

### Explicit Lists
```yaml
temperature: # Explicit values
```

### Tuple Formats
```yaml
# (start, increment) - requires matching value list length
temperature: (300, 50) # 300, 350, 400, ... (length from values)

# (start, stop, step) - step size
temperature: (300, 1000, 10.0) # From 300K to 3000K in steps of 10K

# (start, stop, points) - number of points
temperature: (300, 1000, 71) # 71 evenly spaced points

# Decreasing temperature
temperature: (1000, 300, -5.0) # 1000, 995, 990, ..., 300
```

### Temperature References
```yaml
# Direct references
temperature: melting_temperature
temperature: solidus_temperature

# Arithmetic expressions
temperature: melting_temperature + 50
temperature: liquidus_temperature - 10
```

## Advanced Features

### Regression Configuration
Control data simplification and memory usage:
```yaml
properties:
    heat_conductivity:
      temperature: [1200, 1800, 2200, 2400]  # Temperatures in Kelvin
      value: [25, 30, 33, 35]  # Property values
      bounds: [constant, constant]
      regression:
          simplify: pre # Apply before symbolic processing
          degree: 1 # Linear regression
          segments: 3 # Number of piecewise segments
```

### Boundary Behavior
Control extrapolation outside data range:
```yaml
bounds: [constant, extrapolate]
```
- `constant`: Use boundary values as constants outside range
- `extrapolate`: Linear extrapolation outside range

## Creating a Complete Material Definition

Here's a complete example for stainless steel SS304L:

```yaml
name: Stainless Steel 304L
material_type: alloy  # Must be 'alloy' or 'pure_metal'

# Composition fractions must sum to 1.0
composition:
  Fe: 0.675
  Cr: 0.170
  Ni: 0.120
  Mo: 0.025
  Mn: 0.01

# Required temperature properties for alloys
solidus_temperature: 1605.          # Temperature where melting begins (K)
liquidus_temperature: 1735.         # Temperature where material is completely melted (K)
initial_boiling_temperature: 6520.  # Temperature where boiling begins (K)
final_boiling_temperature: 6850.    # Temperature where material is completely vaporized (K)

properties:

  latent_heat_of_vaporization: 1.71401E5  # J/kg - Scientific notation supported

  latent_heat_of_fusion:
    temperature: [solidus_temperature - 5, liquidus_temperature + 5]  # Temperature references with arithmetic
    value: [0, 171401.]                    # Corresponding property values
    bounds: [constant, constant]           # Boundary behavior: 'constant' or 'extrapolate'

  thermal_expansion_coefficient:
    temperature: [300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1500, 2000, 2500, 3000]
    value: [1.2632e-5, 1.468e-5, 1.524e-5, 1.581e-5, 1.639e-5, 1.699e-5, 1.759e-5, 1.821e-5, 1.885e-5, 2.1e-5, 2.3e-5, 2.5e-5, 2.7e-5]
    bounds: [constant, constant]
    regression:                            # Optional regression configuration
      simplify: post                        # 'pre' (before processing) or 'post' (after processing)
      degree: 1                            # Polynomial degree for regression
      segments: 2                          # Number of piecewise segments

  heat_capacity:
    file_path: ./SS304L.xlsx              # Relative path from YAML file location
    temperature_column: T (K)             # Column name for temperature data
    property_column: Specific heat (J/(Kg K)) # Column name for property data
    bounds: [constant, constant]          # Required boundary behavior
    regression:                           # Optional regression for data simplification
      simplify: pre                       # Apply regression before processing
      degree: 1                           # Linear regression
      segments: 4                         # Divide into 4 piecewise segments

  density:
    file_path: ./SS304L.xlsx
    temperature_column: T (K)
    property_column: Density (kg/(m)^3)
    bounds: [constant, constant]
    regression:
      simplify: post                      # Apply regression after processing
      degree: 1
      segments: 1

  heat_conductivity:
    temperature: [500, 1700, 3000]        # Temperature breakpoints (K)
    equation: [0.0124137215440647*T + 13.0532171803243, 0.0124137215440647*T + 13.0532171803243]
    # Two equations for three breakpoints: [500-1700K] and [1700-3000K]
    bounds: [constant, constant]           # Boundary behavior outside range
    regression:
      simplify: post                      # Apply regression after symbolic processing
      degree: 1
      segments: 2

  specific_enthalpy:
    temperature: (300, 3000, 541)         # (start, stop, num_points) - 541 evenly spaced points
    equation: Integral(heat_capacity, T)   # Symbolic integration
    bounds: [constant, constant]
    regression:
      simplify: post
      degree: 1
      segments: 2

  energy_density:
    temperature: (300, 3000, 5.0)         # (start, stop, step) - 5K increments
    equation: density * specific_enthalpy  # Property dependencies automatically resolved
    bounds: [extrapolate, extrapolate]
    regression:
      simplify: post
      degree: 1
      segments: 6

  thermal_diffusivity:
    temperature: (3000, 300, -5.0)        # (start, stop, negative_step) - decreasing temperature
    equation: heat_conductivity /(density * heat_capacity)  # Multi-property dependency
    bounds: [constant, constant]
    regression:
      simplify: post
      degree: 1
      segments: 3
```

## Best Practices

- Use consistent units throughout your definitions (SI units recommended)
- Document the expected units for each property
- For temperature-dependent properties, cover the full range of temperatures you expect in your simulation
- Validate your property data against experimental values when possible
- All numerical values must use period (.) as decimal separator, not comma
- Interpolation between data points is performed automatically for file-based and key-value properties

## Validation and Error Handling

PyMatLib's new architecture includes comprehensive validation:

- **Composition validation**: Ensures fractions sum to 1.0
- **Temperature validation**: Checks for monotonicity and physical validity
- **Property dependency validation**: Ensures computed properties have required dependencies
- **File validation**: Validates file existence and format
- **Type validation**: Ensures proper data types throughout

## Important Notes

- Properties cannot be defined in multiple ways or multiple times
- Required dependencies for computed properties must be present
- Properties will be computed in the correct order regardless of their position in the file
- Temperature arrays must be monotonic
- Energy density arrays must be monotonic with respect to temperature
