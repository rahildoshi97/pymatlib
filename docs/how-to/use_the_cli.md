# Use the Command-Line Interface

Installing MaterForge puts a `materforge` command on your `PATH`. It exposes the
most common operations - listing the bundled examples, validating a YAML file,
inspecting its metadata, plotting it, evaluating its properties, and checking
fit quality - so you can work with material files without writing any Python.

Every subcommand is a thin wrapper over the [public API](../reference/api.rst);
anything the CLI does is also reachable from Python, and vice versa.

```bash
materforge --help        # list all subcommands
materforge --version     # print the installed version
```

---

## Listing the Bundled Examples

```bash
materforge list
```

```text
Bundled example materials (3):
  1.4301
  Al
  Al2O3
```

These are the reference materials shipped inside the package (see
[Load Bundled Example Materials](load_bundled_materials.md)). Add `--paths` to
print the YAML file location next to each name, handy for copying one as a
starting point for your own config:

```bash
materforge list --paths
```

---

## Validating a File

```bash
materforge validate my_material.yaml
```

Prints `OK: my_material.yaml is valid.` and exits `0` when the file is
structurally sound. On a malformed file it prints the parser's error to stderr
and exits `1`. A path that does not exist is rejected up front as a usage error
(exit `2`).

This is the same command the standalone `materforge-validate` alias now runs.

---

## Inspecting Metadata

`info` reports a material's name, its properties, and how each property is
defined - without fully building the symbolic expressions:

```bash
materforge info my_material.yaml
```

```text
Material: Aluminum
Source:   my_material.yaml
Properties (11):
  - density
  - heat_capacity
  ...
Property types:
  CONSTANT_VALUE: 2
  STEP_FUNCTION: 2
  TABULAR_DATA: 2
  PIECEWISE_EQUATION: 1
  COMPUTED_PROPERTY: 4
```

---

## Plotting

```bash
materforge plot my_material.yaml
```

Builds the material with plotting enabled and writes a property figure to a
`materforge_plots/` directory next to the YAML file. Use `-s/--symbol` to change
the dependency symbol substituted for the YAML placeholder `T` (default `T`):

```bash
materforge plot my_material.yaml --symbol u_C
```

---

## Evaluating at a Value

`evaluate` substitutes a single dependency value into every property and prints
the numeric results:

```bash
materforge evaluate my_material.yaml 500
```

```text
Aluminum @ T = 500.0
  density          2634.48
  heat_capacity    959.489
  ...
```

As with `plot`, `-s/--symbol` sets the dependency symbol:

```bash
materforge evaluate my_material.yaml 500 --symbol u_C
```

---

## Checking Fit Quality

`fit` reports how faithfully each data-backed property (file-import, tabular, or
computed) reproduces the data it was built from - the genuine error for a
regression, ~0 for plain interpolation:

```bash
materforge fit my_material.yaml
```

```text
density: R²=0.999995  RMSE=1.263  MAE=0.8293  max|err|=6.347  (n=541)
energy_density: R²=0.999983  RMSE=1.628e+07  MAE=1.334e+07  max|err|=6.507e+07  (n=541)
heat_capacity: R²=0.969656  RMSE=81.45  MAE=37.71  max|err|=755.6  (n=541)
...
```

Pass a property name to score just that one, and `-s/--symbol` to set the
dependency symbol:

```bash
materforge fit my_material.yaml heat_capacity --symbol u_C
```

Constants, step functions, and piecewise-equation properties are exact
definitions with no source data, so they are not reported; a material with none
of them exits with code `1`. See the
[fit-quality guide](assess_fit_quality.md) for the Python API behind this command.

---

## Verbosity and Exit Codes

By default the CLI prints only its own output. Pass `-v` to surface MaterForge's
informational logs on stderr, or `-vv` for debug detail:

```bash
materforge validate my_material.yaml -v
```

Exit codes follow the usual convention: `0` on success, `1` on a handled error
(bad file, invalid YAML), and `2` for a command-line usage error.

---

## Next Steps

- [Define your own material properties](define_materials.md)
- [Load bundled example materials](load_bundled_materials.md) from Python
- [API reference](../reference/api.rst)
