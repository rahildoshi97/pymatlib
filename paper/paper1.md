---
title: 'PyMatLib: A Python Library for Temperature-Dependent Material Property Processing in Scientific Simulations'
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
date: 16 July 2025
bibliography: paper.bib
---

# Summary

PyMatLib is a Python library that helps scientists and engineers handle materials whose properties change with temperature.
When materials are heated or cooled, their physical characteristics like
thermal conductivity, density, and heat capacity change significantly.
This creates challenges for computer simulations of processes like metal casting, heat treatment, or thermal analysis.

The library provides a simple way to define these temperature-dependent properties using YAML configuration files,
then automatically converts them into mathematical expressions that can be used in scientific simulations.
PyMatLib supports both pure metals and alloys,
handles six different ways of defining properties (from simple constants to complex equations),
and automatically manages dependencies between different material properties.

The library has been designed specifically for high-performance simulations in computational materials science
and heat transfer applications, integrating seamlessly with existing scientific computing workflows.

# Statement of Need

Accurate representation of temperature-dependent material properties is fundamental to computational materials science,
thermal analysis, and multi-physics simulations [@lewis1996finite; @zienkiewicz2013finite].
Traditional approaches often require manual interpolation, custom scripting,
or proprietary software solutions that lack flexibility and reproducibility [@ashby2013materials].
Existing materials databases like NIST [@nist_webbook] and CoolProp [@coolprop] typically provide tabular data
without integrated processing capabilities, forcing researchers to implement ad-hoc solutions for each simulation framework.

While established tools like CoolProp [@coolprop] and NIST databases [@nist_webbook] provide comprehensive thermodynamic data,
they lack integrated symbolic processing and dependency resolution capabilities.
CoolProp focuses on fluid properties with limited solid material support and no symbolic integration,
while NIST WebBook provides tabular data without processing capabilities or dependency management.
CALPHAD databases [@lukas2007computational] require specialized software and lack integration with general-purpose simulation codes.
Custom implementations often lack validation, reproducibility, and standardized interfaces.

PyMatLib addresses these limitations by providing a standardized, extensible framework
that offers unprecedented flexibility in material property definition:

**Flexible Input Methods**: The library supports six different property definition methods
(constant values, step functions, file-based data, key-value pairs, piecewise equations, and computed properties),
allowing users to choose the most appropriate format for their data sources and modeling requirements (\autoref{fig:input_methods}).
File-based data processing leverages pandas [@pandas] for robust handling of various formats
including Excel (.xlsx), CSV, and text files with comprehensive missing value detection, data cleaning, duplicate removal, and validation.

![PyMatLib input flexibility demonstration showing different property definition methods: (a) constant values for simple properties.\label{fig:constant}](figures/constant.png)

![PyMatLib input flexibility demonstration showing different property definition methods: (b) step functions for phase transitions.\label{fig:step function}](figures/step_function.png)

![PyMatLib input flexibility demonstration showing different property definition methods: (c) file properties.\label{fig:file}](figures/file.png)

![PyMatLib input flexibility demonstration showing different property definition methods: (d) key-value pairs for experimental data.\label{fig:key val}](figures/key_val.png)

![PyMatLib input flexibility demonstration showing different property definition methods: (e) piecewise equations for complex relationships.\label{fig:piecewise}](figures/piecewise_equation.png)

![PyMatLib input flexibility demonstration showing different property definition methods: (f) computed properties with automatic dependency resolution.\label{fig:compute}](figures/compute.png)

**Universal Material Support**: The framework supports both pure metals and alloys through a unified interface,
with extensibility for additional material types.
Pure metals use melting/boiling temperatures, while alloys use solidus/liquidus temperature ranges.

**Robust Data Quality Assurance**: Built-in data validation includes duplicate temperature removal,
missing value handling with configurable thresholds, automatic data sorting,
and support for various file encodings and missing value representations commonly found in experimental datasets.
This ensures data integrity and prevents common errors in materials property processing.

**Optional Data Simplification**: The library supports regression-based data simplification using pwlf [@pwlf] for large complex datasets,
allowing users to reduce computational overhead and memory usage while maintaining accuracy.
Piecewise linear fitting with configurable polynomial degrees enables efficient approximation of complex property relationships
while preserving essential physical behavior (\autoref{fig:regression_options}).

**Intelligent Simplification Timing**: PyMatLib provides sophisticated control over when data simplification occurs in the dependency chain
through the `simplify` parameter (`pre` or `post`). When set to `pre`, properties are simplified before being passed to dependent properties,
optimizing computational performance for complex dependency networks.
When set to `post`, the raw piecewise function with all data points is preserved during dependency resolution,
with simplification applied only after all dependent calculations are complete.
This ensures maximum accuracy in interdependent property calculations while still providing the benefits of data simplification.
This timing control allows users to balance computational efficiency with numerical accuracy based on their specific simulation requirements.

![Regression capabilities showing data simplification effects: raw experimental data (points) fitted with different polynomial degrees and segment configurations, demonstrating how PyMatLib can reduce complexity while maintaining physical accuracy.\label{fig:regression_options}](figures/regression_options.png)

**Configurable Boundary Behavior**: Users can specify how properties behave outside defined temperature ranges,
choosing between constant extrapolation or linear extrapolation based on their physical understanding of the material
(\autoref{fig:boundary_behavior}).

![Boundary behavior options in PyMatLib showing the same thermal conductivity property with different extrapolation settings: constant boundaries (left) maintain edge values outside the defined range, while extrapolate boundaries (right) use linear extrapolation.\label{fig:boundary_behavior}](figures/boundary_behavior.png)

**Automatic Dependency Resolution**: Intelligent processing order determination for computed properties ensures
mathematical dependencies are resolved correctly without manual intervention.
The library automatically detects circular dependencies and provides clear error messages for invalid configurations.

**Bidirectional Property-Temperature Conversion**: The library automatically generates inverse piecewise functions enabling
temperature = f(property) calculations alongside the standard property = f(temperature) relationships.
This bidirectional capability is essential for energy-based numerical methods [@voller1987fixed], phase-change simulations,
and iterative solvers where temperature must be determined from known property values.
The inverse function generation supports linear piecewise segments
(either through default linear interpolation or explicit degree=1 regression), ensuring robust mathematical invertibility.

**Built-in Validation Framework**: Built-in validation ensures YAML configuration correctness, including composition sum verification,
required field checking for pure metals versus alloys, property name validation, and regression parameter bounds checking.
This prevents common configuration errors and ensures reproducible material definitions [@roache1998verification].

**Integrated Visualization**: Automatic plot generation enables users to verify their material definitions visually,
with the option to disable visualization for production workflows after validation.

Unlike existing tools, PyMatLib uniquely combines symbolic mathematics [@sympy], automatic dependency resolution,
and seamless integration with scientific computing workflows [@numpy; @matplotlib].
The library integrates directly with simulation frameworks like
pystencils [@pystencils] and waLBerla [@walberla] for high-performance computing applications.

The library can be applied in research projects involving alloy design
and optimization with accurate representation of solidus-liquidus temperature ranges [@callister2018materials],
energy-based finite element methods for thermal analysis [@hughes2012finite],
multiscale simulations linking molecular dynamics with continuum mechanics [@tadmor2011modeling],
and high-performance computing applications in computational fluid dynamics and heat transfer modeling.
Its YAML-based configuration system, powered by ruamel.yaml parsing [@ruamel-yaml], supports the FAIR principles [@wilkinson2016fair] for scientific data management,
enabling reproducible research across different simulation codes and research groups.

# Acknowledgements

The development of PyMatLib was supported by the Friedrich-Alexander-Universität Erlangen-Nürnberg.
We acknowledge the contributions of the SymPy [@sympy], NumPy [@numpy], pandas [@pandas], matplotlib [@matplotlib],
and ruamel.yaml [@ruamel-yaml] development communities, whose libraries provide
the symbolic mathematics, numerical computing, data processing, visualization, and configuration parsing capabilities
that form the foundation of PyMatLib.

# References
