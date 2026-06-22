# CouetteFlow benchmark

A 3D thermal Couette flow benchmark integrating MaterForge with
[waLBerla](https://i10git.cs.fau.de/walberla/walberla),
[lbmpy](https://pypi.org/project/lbmpy/), and
[pystencils](https://pypi.org/project/pystencils/).

It demonstrates a MaterForge-driven temperature-dependent viscosity `nu(T)` baked
into a generated LBM sweep, and can also bake in a constant viscosity for
comparison.

> **Build system:** this app is built through the top-level
> [`apps/CMakeLists.txt`](../CMakeLists.txt) driver, which adds the shared
> `walberla/` submodule once and configures both demo apps. Configure from
> `apps/`; the binary lands in `apps/build/<preset>/`. See
> [`apps/README.md`](../README.md) for the overview and prerequisites.

## License

All files under `apps/` (except the `walberla/` submodule) are licensed under
**GPL-3.0-or-later** (inherited from waLBerla). See
[`apps/LICENSE`](../LICENSE). MaterForge itself (`../../src/materforge/`) is
licensed under **BSD-3-Clause** - see the repository root for details.

## Files

| File | Purpose |
|------|---------|
| `CouetteFlowScaling.cpp`  | LBM driver |
| `CouetteFlowSweeps.py`    | pystencils/lbmpy code generator |
| `CouetteFlowMaterial.yaml`| MaterForge material spec |
| `CouetteFlowScaling.prm`  | runtime parameters |
| `CMakeLists.txt`          | app target + MaterForge viscosity options |

## Tuning the viscosity fit (MaterForge ≥ 0.10.0)

`dynamic_viscosity` in `CouetteFlowMaterial.yaml` carries a `regression:` block
(`simplify`, `degree`, `segments`) that fits its smooth analytic form to a cheap
piecewise polynomial. With `USE_MATERFORGE=ON`, code generation reports how
faithfully that fit reproduces the source data, via the MaterForge fit-quality
API, e.g.:

```
Fit quality: dynamic_viscosity: R²=0.999989  RMSE=0.0001226  MAE=0.0001085  max|err|=0.0002464  (n=31)
```

To choose the cheapest `(segments, degree)` that meets an accuracy target, pass
`--survey-viscosity-fit` to the code generator. It rebuilds only the viscosity
for a small grid of configs, scores each with `materforge.fit_quality`, and
recommends the lowest-cost config under tolerance (it only prints - the generated
code is unchanged):

```
Viscosity fit survey - cheapest (segments, degree) with max|err| < 0.001:
segments degree        R^2        RMSE    max|err|   ok
       1      1   0.969131   6.397e-03   1.466e-02   no
       2      1   0.998113   1.582e-03   3.258e-03   no
       1      2   0.999584   7.427e-04   1.761e-03   no
       2      2   0.999989   1.226e-04   2.464e-04  yes
       3      2   0.999999   4.112e-05   7.050e-05  yes
       2      3   1.000000   3.899e-06   8.571e-06  yes
  -> use segments=2, degree=2 (max|err|=2.46e-04); set this in the regression block.
```

## Configure

The build is driven from `apps/` by a small set of CMake cache variables. The
`apps/CMakePresets.json` / `apps/CMakeUserPresets.json` files provide defaults
for a typical workstation, the woody cluster, and LUMI; pass extra `-D` flags
after the preset name to override individual options.

### Cache variables

| Variable            | Default     | Purpose                                                       |
|---------------------|-------------|---------------------------------------------------------------|
| `TARGET_PLATFORM`   | `CPU`       | `CPU`, `GPU-CUDA`, or `GPU-HIP` (drives the shared waLBerla build) |
| `USE_MATERFORGE`    | `ON`        | `ON` -> MaterForge `nu(T)` baked in; `OFF` -> literal `CONST_NU` |
| `CONST_NU`          | `0.16667`   | Constant kinematic viscosity baked into the sweep when `USE_MATERFORGE=OFF` |
| `COLLISION_OP`      | `SRT`       | LBM collision operator: `SRT`, `TRT`, or `MRT`               |
| `WRITE_VISCOSITY`   | `OFF`       | `ON` -> write the `nu` field each step for VTK; `OFF` -> omit it. Sets the generated `StreamCollide` constructor arity, which `CouetteFlowScaling.cpp` matches automatically. |

### Available presets

| Preset                | Platform     | Build type | Notes                                |
|-----------------------|--------------|------------|--------------------------------------|
| `local-debug-cpu`     | CPU          | Debug      | local workstation                    |
| `local-release-cpu`   | CPU          | Release    | local workstation                    |
| `local-debug-gpu`     | GPU-CUDA     | Debug      | local workstation with CUDA          |
| `local-release-gpu`   | GPU-CUDA     | Release    | local workstation with CUDA          |
| `woody-release-cpu`   | CPU          | Release    | Xeon Gold 6326 (Ice Lake) on woody   |
| `lumi-release-cpu`    | CPU          | Release    | LUMI (Cray CCE)                      |
| `lumi-release-gpu`    | GPU-HIP      | Release    | LUMI-G (ROCm, gfx90a)                |
| `lumi-debug-gpu`      | GPU-HIP      | Debug      | LUMI-G (ROCm, gfx90a)                |

## Build

```bash
cd apps/

# Configure + build (CPU release). Use -DBUILD_HEAT_EQUATION=OFF to build only this app.
cmake --preset local-release-cpu
cmake --build --preset local-release-cpu-build

# Constant viscosity nu=0.1 baked at compile time
cmake --preset local-release-cpu -DUSE_MATERFORGE=OFF -DCONST_NU=0.1
cmake --build --preset local-release-cpu-build
```

The single binary works on the configured platform; the `CouetteFlowSweeps.py`
sweep is regenerated automatically when its inputs change. The executable is
written to `apps/build/<preset>/CouetteFlowScaling`.

## Run

```bash
# CWD = apps/CouetteFlow/ so the VTK output path "output/vtk" (relative)
# resolves into apps/CouetteFlow/output/vtk/.
cd apps/CouetteFlow/
../build/local-release-cpu/CouetteFlowScaling CouetteFlowScaling.prm
```

The binary prints **MLUPS per process** and **total MLUPS** at the end of the
timed run.

### Command-line overrides

`.prm` parameters can be overridden using waLBerla's `-<Block>.<Key>=<Value>`
syntax (the `=` is required). Vector3 values use the angle-bracket form
`<x,y,z>`:

```bash
../build/local-release-cpu/CouetteFlowScaling CouetteFlowScaling.prm \
    -DomainSetup.blocks="<4,1,1>" \
    -DomainSetup.cellsPerBlock="<64,32,32>" \
    -Parameters.timesteps=10000 \
    -Output.vtkWriteFrequency=200 \
    -Output.vtkOutputDir=output/vtk
```

`Output.vtkWriteFrequency=0` disables VTK output; any non-zero value enables it
with the given write frequency.

### `.prm` reference (`CouetteFlowScaling.prm`)

| Block         | Key                | Default       | Purpose                                          |
|---------------|--------------------|---------------|--------------------------------------------------|
| `DomainSetup` | `blocks`           | `<1,2,2>`     | MPI process grid (Vector3)                       |
| `DomainSetup` | `cellsPerBlock`    | `<128,32,32>` | Cells per block (Vector3)                        |
| `DomainSetup` | `periodic`         | `<1,1,0>`     | Per-axis periodicity                             |
| `Parameters`  | `nu`               | `0.08`        | Initial fill / VTK label only (physics baked in) |
| `Parameters`  | `u_max`            | `0.025`       | Wall velocity (lattice units)                    |
| `Parameters`  | `timesteps`        | `60000`       | Total simulation steps                           |
| `Parameters`  | `errorThreshold`   | `1e-3`        | Convergence tolerance for steady-state check     |
| `Parameters`  | `T_bottom`         | `300.0`       | Bottom-wall temperature (K)                      |
| `Parameters`  | `T_top`            | `600.0`       | Top-wall temperature (K)                         |
| `Output`      | `vtkWriteFrequency`| `0`           | Steps between VTK writes (0 = disabled)          |
| `Output`      | `vtkOutputDir`     | `output/vtk`  | VTK output directory (relative to CWD)           |

> When `USE_MATERFORGE=OFF`, `Parameters.nu` must match the `CONST_NU` you built
> with - the `nu` field is only used for the VTK output filename and for the
> initial fill of the viscosity scalar field. The physics viscosity is the
> literal baked into the generated sweep.
