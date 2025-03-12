# Material Properties in pymatlib

This document explains the conceptual framework behind temperature-dependent material properties in pymatlib,
how they are represented internally, and the mathematical models used for computed properties.

## Conceptual Framework

Material properties in pymatlib are designed around these key principles:

1. **Temperature Dependence**: Most material properties vary with temperature, especially during phase transitions
2. **Symbolic Representation**: Properties are represented as symbolic expressions for mathematical manipulation
3. **Flexible Definition**: Properties can be defined through various methods (constants, data points, files, or computation)
4. **Physical Consistency**: Computed properties follow established physical relationships

## Internal Representation

### MaterialProperty Class

At the core of pymatlib's property system is the `MaterialProperty` class, which contains:

- A symbolic expression (`expr`) representing the property
- A list of assignments needed to evaluate the expression
- Methods to evaluate the property at specific temperatures

```python
@dataclass
class MaterialProperty:
    expr: sp.Expr
    assignments: List[Assignment] = field(default_factory=list)
    
    def evalf(self, symbol: sp.Symbol, temperature: Union[float, ArrayTypes]) -> Union[float, np.ndarray]:
        # Evaluates the property at the given temperature
```

### Assignment Class

The `Assignment` class represents intermediate calculations needed for property evaluations:

```python
@dataclass
class Assignment:
    lhs: sp.Symbol
    rhs: Union[tuple, sp.Expr]
    lhs_type: str
```

## Property Definition Methods

pymatlib supports multiple ways to define material properties:

1. Constant Values

Properties that don't vary with temperature are defined as simple numeric values:

```yaml
thermal_expansion_coefficient: 16.3e-6
```

Internally, these are converted to constant symbolic expressions.

2. Interpolated Values

For properties that vary with temperature, pymatlib supports interpolation between data points:

```yaml
heat_conductivity:
    key: [1200, 1800, 2200, 2400]  # Temperatures in Kelvin
    val: [25, 30, 33, 35]          # Property values
```

Internally, these are represented as piecewise functions that perform linear interpolation between points.

3. File-Based Properties
   
Properties can be loaded from external data files:

```yaml
density:
    file: ./material_data.xlsx
    temp_col: T (K)
    prop_col: Density (kg/(m)^3)
```

The data is loaded and converted to an interpolated function similar to key-value pairs.

4. Computed Properties

Some properties can be derived from others using physical relationships:

```yaml
thermal_diffusivity: compute  # k/(ρ*cp)
```

## Computed Models

pymatlib implements several physical models for computing properties:

### Density by Thermal Expansion

```text
ρ(T) = ρ₀ / (1 + tec * (T - T₀))³
```

Where:
- ρ₀ is the base density
- T₀ is the base temperature
- tec is the thermal expansion coefficient

### Thermal Diffusivity

```text
α(T) = k(T) / (ρ(T) * cp(T))
```

Where:
- k(T) is the thermal conductivity
- ρ(T) is the density
- cp(T) is the specific heat capacity

### Energy Density

pymatlib supports multiple models for energy density:

1. **Standard Model**

```text
E(T) = ρ(T) * (cp(T) * T + L)
```

2. **Enthalpy-Based Model**

```text
E(T) = ρ(T) * (h(T) + L)
```

3. **Total Enthalpy Model**
```text
E(T) = ρ(T) * h(T)
```

Where:
- ρ(T) is the density
- cp(T) is the specific heat capacity
- h(T) is the specific enthalpy
- L is the latent heat of fusion

## Interpolation Between Data Points

For properties defined through key-value pairs or files, pymatlib performs linear interpolation between data points:

1. For a temperature T between two known points T₁ and T₂:

```text
property(T) = property(T₁) + (property(T₂) - property(T₁)) * (T - T₁) / (T₂ - T₁)
```

2. For temperatures outside the defined range, the property value is clamped to the nearest endpoint.

## Temperature Arrays for EnergyDensity

When using computed energy density, you must specify a temperature array:

```text
energy_density_temperature_array: (300, 3000, 541)  # 541 points from 300K to 3000K
```

This array is used to pre-compute energy density values for efficient interpolation during simulations.

### Best Practices

1. **Use Consistent Units**: All properties should use SI units (m, s, kg, K, etc.)
2. **Cover the Full Temperature Range**: Ensure properties are defined across the entire temperature range of your simulation
3. **Add Extra Points Around Phase Transitions**: For accurate modeling, use more data points around phase transitions
4. **Validate Against Experimental Data**: When possible, compare property values with experimental measurements
5. **Document Property Sources**: Keep track of where property data comes from for reproducibility
