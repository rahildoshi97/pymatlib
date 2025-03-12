# Defining Custom Material Properties

This guide explains how to define custom material properties in pymatlib using different methods.

## YAML Configuration Options

pymatlib supports several ways to define material properties in YAML files, following SI units (m, s, kg, A, V, K, etc.).

### 1. Constant Value

For properties that don't vary with temperature:

```yaml
properties:
    thermal_expansion_coefficient: 16.3e-6
```

### 2. Key-Value Pairs for Interpolation

For properties that vary with temperature:

```yaml
properties:
    # Using explicit temperature list
    heat_conductivity:
        key: [1200, 1800, 2200, 2400]  # Temperatures in Kelvin
        val: [25, 30, 33, 35]  # Property values
    
    # Using references to defined temperatures
    latent_heat_of_fusion:
        key: [solidus_temperature, liquidus_temperature]
        val: [171401, 0]
    
    # Using tuple for temperature generation
    heat_capacity:
        key: (1000, 200)  # Start at 1000K and increment by 200K for each value in val
        # Generates: [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000, 3200]
        val: [580, 590, 600, 600, 600, 610, 620, 630, 660, 700, 750, 750]
    
    # Using tuple with negative increment
    density:
        key: (1735.00, -5)  # Start at 1735.00K and decrement by 5K for each value in val
        # Generates: [1735.00, 1730.00, 1725.00, 1720.00, 1715.00, 1710.00, 1705.00, 1700.00, 1695.00, 1690.00]
        val: [7037.470, 7060.150, 7088.800, 7110.460, 7127.680, 7141.620, 7156.800, 7172.590, 7184.010, 7192.780]
```

### 3. Loading from External Files

For properties defined in spreadsheets:

```yaml
properties:
    # Simple format (first column = temperature, second column = property)
    heat_capacity: ./heat_capacity_data.txt
    
    # Advanced format (specify columns)
    density:
      file: ./304L_data.xlsx
      temp_col: T (K)
      prop_col: Density (kg/(m)^3)
```

Supported file formats include .txt (space/tab separated), .csv, and .xlsx.

### 4. Energy density temperature arrays

For properties that need to be evaluated at specific temperature points:

```yaml
properties:
    # Using count (integer)
    energy_density_temperature_array: (300, 3000, 541)  # 541 evenly spaced points
    # OR
    # Using step size (float)
    energy_density_temperature_array: (300, 3000, 5.0)  # From 300K to 3000K in steps of 5K
    # OR
    # Descending order
    energy_density_temperature_array: (3000, 300, -5.0)  # From 3000K to 300K in steps of -5K
```

### 5. Computed Properties

For properties that can be derived from others:

```yaml
properties:
    # Simple format for density
    density: compute

    # Simple format for thermal_diffusivity
    thermal_diffusivity: compute # Will be calculated from k/(ρ*cp)

    # Simple format for energy_density
    energy_density: compute  # Uses default model: ρ(T) * (cp(T) * T + L)
    # OR
    # Advanced format with explicit model selection for energy_density
    energy_density:
        compute: enthalpy_based  # Uses model: ρ(T) * (h(T) + L)
    # OR
    energy_density:
        compute: total_enthalpy  # Uses model: ρ(T) * h(T)
```

### The equations used for computed properties are:
- Density by thermal expansion: ρ(T) = ρ₀ / (1 + tec * (T - T₀))³
- - Required: base_temperature, base_density, thermal_expansion_coefficient

- Thermal diffusivity: α(T) = k(T) / (ρ(T) * cp(T))
- - Required: heat_conductivity, density, heat_capacity

- Energy density (standard model): ρ(T) * (cp(T) * T + L)
- - Required: density, heat_capacity, latent_heat_of_fusion

- Energy density (enthalpy_based): ρ(T) * (h(T) + L)
- - Required: density, specific_enthalpy, latent_heat_of_fusion

- Energy density (total_enthalpy): ρ(T) * h(T)
- - Required: density, specific_enthalpy

## Creating a Complete Alloy Definition

Here's a complete example for stainless steel [SS304L](https://i10git.cs.fau.de/rahil.doshi/pymatlib/-/blob/master/src/pymatlib/data/alloys/SS304L/SS304L.yaml?ref_type=heads):

```yaml
name: SS304L

composition:
    C: 0.0002
    Si: 0.0041
    Mn: 0.016
    P: 0.00028
    S: 0.00002
    Cr: 0.1909
    N: 0.00095
    Ni: 0.0806
    Fe: 0.70695

solidus_temperature: 1605
liquidus_temperature: 1735

properties:
    energy_density:
      compute: total_enthalpy
    energy_density_temperature_array: (300, 3000, 541)

    base_temperature: 2273.
    base_density: 6.591878918e3

    density:
      file: ./304L_data.xlsx
      temp_col: T (K)
      prop_col: Density (kg/(m)^3)

    heat_conductivity:
      file: ./304L_data.xlsx
      temp_col: T (K)
      prop_col: Thermal conductivity (W/(m*K))

      heat_capacity:
        file: ./304L_data.xlsx
        temp_col: T (K)
        prop_col: Specific heat (J/(Kg K))

      thermal_expansion_coefficient: 16.3e-6
    
      specific_enthalpy:
        file: ./304L_data.xlsx
        temp_col: T (K)
        prop_col: Enthalpy (J/kg)
    
      latent_heat_of_fusion:
        key: [solidus_temperature, liquidus_temperature]
        val: [171401, 0]
    
      thermal_diffusivity: compute
```

## Best Practices

- Use consistent units throughout your definitions
- Document the expected units for each property
- For temperature-dependent properties, cover the full range of temperatures you expect in your simulation
- Validate your property data against experimental values when possible
- Use computed properties only when the relationship is well-established
- All numerical values must use period (.) as decimal separator, not comma
- Interpolation between data points is performed automatically for file-based and key-val properties

## Important Notes

- If a specific property is defined in multiple ways or multiple times, the parser will throw an error
- If required dependencies for computed properties are missing, an error will be raised
- Properties will be computed in the correct order regardless of their position in the file
- To retrieve temperature from energy_density, use the default "interpolate" method from within the generated class from InterpolationArrayContainer named after the alloy
