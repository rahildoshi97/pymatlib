---
title: 'SymMatsPy: Symbolic Materials for Scientific Simulations in Python'
tags:
  - Python
  - materials science
  - computational physics
  - thermodynamics
  - finite element analysis
  - material properties
authors:
  - name: Rahil Miten Doshi
    orcid: 0009-0008-3570-9841
    affiliation: 1
affiliations:
  - name: Friedrich-Alexander-Universität Erlangen-Nürnberg, Germany
    index: 1
date: 24 July 2025
bibliography: paper.bib
---

# Summary

PyMatLib is an extensible, open-source Python library that streamlines the definition and use of 
material properties in numerical simulations.
The library allows researchers to define complex material behaviors in simple human-readable YAML files,
which are automatically converted into symbolic mathematical expressions for direct use in scientific computing frameworks. 
PyMatLib supports both pure metals and alloys, 
offers six different property definition methods, 
and intelligently manages dependencies between different material properties.
It is designed for high-performance computing applications,
and serves as a seamless bridge between experimental data and numerical simulation.

# Statement of Need

The physical characteristics of materials, such as thermal conductivity, density,
and heat capacity, in fields like metal casting, heat treatment, and thermal analysis, change significantly with temperature.
Accurately modeling these changes is a persistent challenge.

Material properties are not constants—they are typically dependent on variables such as 
temperature, pressure, or concentration, which creates significant challenges for accurate numerical simulations in fields like 
metal casting, heat treatment, and thermal analysis [@lewis1996finite; @zienkiewicz2013finite]. 
The knowledge base for material properties varies dramatically across different materials: 
some well-established materials have been characterized over decades with extensive databases, 
while newly designed alloys from ongoing materials development may have limited or no available data. 
This diversity in material knowledge means that property definitions can range from 
simple constants to complex measurement datasets to sophisticated theoretical models—essentially, 
any representation is possible depending on the material and application.

Researchers often rely on manual interpolation, custom scripting, 
or proprietary software solutions, which can lead to issues with reproducibility, flexibility and standardization [@ashby2013materials]. 
While valuable resources like NIST [@nist_webbook] and libraries like CoolProp [@coolprop] provide extensive material data,
they primarily offer raw tabular data or focus on fluid properties, lacking the integrated symbolic processing and dependency management
capabilities needed for complex simulations. 
Similarly, specialized CALPHAD databases [@calphad] require proprietary software and do not easily integrate with general-purpose simulation codes.

This gap forces researchers to develop ad-hoc solutions for each new project, 
creating a bottleneck in the research workflow and hindering the FAIR principles of data sharing [@wilkinson2016fair]. 
PyMatLib addresses these challenges by combining all these diverse material property representations into a unified material library 
through symbolic mathematical expressions with automatic regression and optimization capabilities.
The library provides a robust, flexible, and open-source tool 
that combines a user-friendly configuration system with powerful backend processing, 
thereby standardizing and simplifying the integration of realistic material behavior into scientific simulations.

# Key Functionality

- **Flexible Input Methods**: The library supports six different property definition methods: 
constant values, step functions, file-based data (Excel, CSV, txt), tabular data, piecewise equations, and computed properties (\autoref{fig:input_methods}). 
This versatility allows users to leverage data from diverse sources, from simple constants to complex experimental datasets. 
File processing is handled robustly using pandas [@pandas].

![PyMatLib's property definition methods: (a) constant value, (b) step function, (c) file data, (d) tabular data, (e) piecewise equations, and (f) computed properties.\label{fig:input_methods}](figures/input_methods.png)

- **Universal Material Support**: The framework is designed to support any material type through its extensible architecture.
Currently, PyMatLib is implemented and thoroughly tested for pure metals and alloys through a unified interface.
The modular design allows for straightforward extension to additional material classes such as ceramics, polymers, composites,
or other specialized materials as research evolves.

- **Automatic Dependency Resolution**: For properties that depend on others (e.g., thermal diffusivity calculated from thermal conductivity, density, and heat capacity), 
PyMatLib automatically determines the correct processing order and resolves mathematical dependencies without manual intervention. 
The library detects circular dependencies and provides clear error messages for invalid configurations, 
- freeing users from complex dependency management.

- **Regression and Data Reduction**: The library integrates pwlf [@pwlf] to perform piecewise linear regression for large datasets.
This simplifies complex property curves into efficient, accurate mathematical representations with configurable polynomial degrees and segments, 
reducing computational overhead while maintaining physical accuracy (\autoref{fig:regression_options}).

![Regression capabilities showing data simplification effects: raw experimental data (points) fitted with different polynomial degrees and segment configurations, demonstrating how PyMatLib can reduce complexity while maintaining physical accuracy.\label{fig:regression_options}](figures/regression_options.png)

- **Intelligent Simplification Timing**: PyMatLib provides sophisticated control over when data simplification occurs
  in the dependency chain through the `simplify` parameter. With `simplify: pre`,
  properties are simplified using regression before being used in dependent calculations, optimizing performance.
  With `simplify: post`, simplification is deferred until all dependent properties have been computed, maximizing numerical accuracy.
  This timing control allows users to balance computational efficiency with numerical accuracy based on their specific simulation requirements.

```yaml
    regression:      # Optional regression configuration
      simplify: pre  # 'pre' (before processing) or 'post' (after processing)
      degree: 2      # Polynomial degree for regression
      segments: 3    # Number of piecewise segments
```

- **Configurable Boundary Behavior**: Users can define how properties behave outside their specified temperature ranges, 
choosing between constant value or linear extrapolation to best match the physical behavior of the material 
(\autoref{fig:boundary_behavior}).
```yaml
    bounds: [constant, extrapolate]  # Boundary behavior: 'constant' or 'extrapolate'
```

![Boundary behavior options in PyMatLib showing the same density property with different extrapolation settings: constant boundaries (left) maintain edge values outside the defined range, while extrapolate boundaries (right) use linear extrapolation.\label{fig:boundary_behavior}](figures/boundary_behavior.png)

- **Automatic Dependency Resolution**: Intelligent processing order determination for computed properties ensures 
mathematical dependencies are resolved correctly without manual intervention. 
The library automatically detects circular dependencies and provides clear error messages for invalid configurations.

- **Bidirectional Property-Variable Inversion**: The library can automatically generate inverse piecewise functions, 
enabling the determination of independent variables from known property values (e.g., `temperature = f(property)`). 
This bidirectional capability is essential for energy-based numerical methods [@voller1987fixed], phase-change simulations, 
and iterative solvers where the independent variable must be determined from known property states. 
Currently, PyMatLib focuses on temperature-dependent properties, 
but the underlying architecture is designed to accommodate additional dependencies such as concentration, pressure, or shear rate in future versions. 
The inverse function generation supports linear piecewise segments 
(either through default linear interpolation or explicit `degree=1` regression), 
ensuring robust mathematical invertibility for the supported variable relationships.

- **Built-in Validation Framework**: A comprehensive validation framework checks YAML configurations for correctness,
including composition sums, required fields for pure metals versus alloys, and valid property names.
This prevents common configuration errors and ensures reproducible material definitions [@roache1998verification].

- **Integrated Visualization**: An integrated visualization tool using matplotlib [@matplotlib]
allows users to automatically generate plots to verify their property definitions visually,
with the option to disable visualization for production workflows after validation.

# Usage Example

PyMatLib is designed for ease of use. A material is defined in a YAML file and loaded with a single function call.
The YAML files can include pure metals with melting/boiling temperatures or alloys with solidus/liquidus temperature ranges.

## YAML Configuration Examples

### Pure Metal (`Al.yaml`)
```yaml
name: Aluminum
material_type: pure_metal

# Composition must sum to 1.0 (for pure metals, single element = 1.0)
composition:
  Al: 1.0  # Aluminum

# Required temperature properties for pure metals
melting_temperature: 933.47  # Solid becomes liquid (K)
boiling_temperature: 2743.0  # Liquid becomes gas (K)

properties:
  thermal_expansion_coefficient:
    temperature: [373.15, 473.15, 573.15, 673.15, 773.15, 873.15]
    value: [2.38e-05, 2.55e-05, 2.75e-05, 2.95e-05, 3.15e-05, 3.35e-05]
    bounds: [constant, constant]
    regression:
      simplify: post
      degree: 1
      segments: 1

  density:
    temperature: (300, 3000, 541)
    equation: 2700 / (1 + 3 * thermal_expansion_coefficient * (T - 293.15))
    bounds: [constant, constant]
    regression:
      simplify: pre
      degree: 1
      segments: 1
```

### Alloy (`SS304L.yaml`)
```yaml
name: Stainless Steel 304L
material_type: alloy

# Composition fractions must sum to 1.0
composition:
  Fe: 0.675  # Iron
  Cr: 0.170  # Chromium
  Ni: 0.120  # Nickel
  Mo: 0.025  # Molybdenum
  Mn: 0.010  # Manganese

# Required temperature properties for alloys
solidus_temperature: 1605.          # Melting begins (K)
liquidus_temperature: 1735.         # Material is completely melted (K)
initial_boiling_temperature: 3090.  # Boiling begins (K)
final_boiling_temperature: 3200.    # Material is completely vaporized (K)

properties:
  density:
    file_path: ./SS304L.xlsx
    temperature_header: Temperature (K)
    value_header: Density (kg/(m)^3)
    bounds: [constant, extrapolate]
    regression:      # Optional regression configuration
      simplify: pre  # Simplify before processing
      degree: 2      # Use quadratic regression for simplification
      segments: 3    # Fit with 3 segments for piecewise linear approximation
```
Complete YAML configurations for both are provided in the PyMatLib [documentation](https://github.com/rahildoshi97/pymatlib/blob/master/docs/how-to/define_materials.md). 

**Python Usage**:

Integrating PyMatLib into a scientific workflow is straightforward.
The primary entry point is the create_material function, which parses the YAML file and returns a fully configured material object.
```python
    import sympy as sp
    from pymatlib.parsing.api import create_material

    # Create a material with a symbolic temperature variable
    T = sp.Symbol('T')
    aluminum = create_material('Al.yaml', T, enable_plotting=True)

    # Access properties as symbolic expressions
    print(f"Density: {aluminum.density}")
    # Output: Piecewise((2678.43051234161, u_C < 300.0), 
    #                   (2744.36352618972 - 0.21977671282703*u_C, u_C < 3000.0), 
    #                   (2085.03338770863, True))

    # Evaluate properties at a specific temperature
    density_at_500K = aluminum.density.subs(T, 500).evalf()
    print(f"Density at 500 K: {density_at_500K:.2f} kg/m^3")
    # Output: Density at 500 K: 2634.48 kg/m^3
```

# Comparison with Existing Tools

| Feature                  | **PyMatLib**    | **CoolProp** | **NIST WebBook** | **CALPHAD** |
|:-------------------------|:----------------|:-------------|:-----------------|:------------|
| **Core Capabilities**    |                 |              |                  |             |
| Symbolic Integration     | Yes             | No           | No               | No          |
| Dependency Resolution    | Yes (Automatic) | No           | No               | No          |
| Multiple Input Methods   | Yes (6 types)   | No           | No               | No          |
|                          |                 |              |                  |             |
| **Material Support**     |                 |              |                  |             |
| Solid Materials          | Yes             | Limited      | Yes              | Yes         |
| Custom Properties        | Yes             | No           | No               | Limited     |
| Temperature Dependencies | Yes             | Yes          | Yes              | Yes         |
|                          |                 |              |                  |             |
| **Accessibility**        |                 |              |                  |             |
| Open Source              | Yes             | Yes          | No               | No          |
| Python Integration       | Native          | Yes          | API only         | No          |

**Key Advantage**: PyMatLib's unique combination of native symbolic mathematics via SymPy [@sympy], 
automatic dependency resolution, and multiple input methods provides a level of flexibility and integration 
not found in existing tools, enabling more reproducible and sophisticated scientific simulations.

# Research Applications and Availability

PyMatLib is applicable to a wide range of research areas, including alloy design and optimization [@callister2018materials], 
energy-based finite element methods for thermal analysis [@hughes2012finite], multiscale simulations [@tadmor2011modeling], 
and high-performance computing in fluid dynamics and heat transfer.
Its architecture promotes reproducible science and is well-suited for high-performance computing environments, 
with demonstrated integrations into frameworks like pystencils [@pystencils] and waLBerla [@walberla].

PyMatLib is an open-source software distributed under the [BSD-3-Clause License](https://github.com/rahildoshi97/pymatlib/blob/master/LICENSE). 
The source code, documentation, and further examples are available on
[GitHub](https://github.com/rahildoshi97/pymatlib/tree/master).

# Acknowledgements

The development of PyMatLib was supported by the Friedrich-Alexander-Universität Erlangen-Nürnberg. 
We acknowledge the developers of SymPy [@sympy], NumPy [@numpy], pandas [@pandas], matplotlib [@matplotlib],
and ruamel.yaml [@ruamel-yaml], whose libraries provide 
the symbolic mathematics, numerical computing, data processing, visualization, and configuration parsing capabilities 
that form the foundation of PyMatLib.

# References
