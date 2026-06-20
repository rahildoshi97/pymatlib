# MaterForge demonstration apps

Two demonstration applications that integrate MaterForge with
[waLBerla](https://i10git.cs.fau.de/walberla/walberla),
[lbmpy](https://pypi.org/project/lbmpy/), and
[pystencils](https://pypi.org/project/pystencils/) via code generation.

| App | What it shows | Detail |
|-----|---------------|--------|
| [`HeatEquationKernel/`](HeatEquationKernel/README.md) | Minimal end-to-end example: a 2D transient heat-equation solver with a MaterForge temperature-dependent thermal diffusivity (1.4301 stainless steel). | [README](HeatEquationKernel/README.md) |
| [`CouetteFlow/`](CouetteFlow/README.md) | A 3D thermal Couette flow LBM benchmark with a MaterForge `nu(T)` baked into the generated sweep (or a constant for comparison). | [README](CouetteFlow/README.md) |

> **Not on PyPI.** These apps are **not** part of the `materforge` wheel. The
> published package is BSD-3-Clause and pip-installable; these apps are
> GPL-3.0-or-later (they link waLBerla/pystencils) and need a C++/CMake/MPI
> toolchain that pip cannot provide. They live only in the source repository -
> get them with `git clone` + submodule init (below).

## License

Everything under `apps/` (except the `walberla/` submodule) is licensed under
**GPL-3.0-or-later**, inherited from waLBerla. See [LICENSE](LICENSE).
MaterForge itself (`../src/materforge/`) is **BSD-3-Clause** - see the
repository root.

The `walberla/` directory is a Git submodule containing waLBerla (GPLv3),
shared by both apps, maintained at
<https://i10git.cs.fau.de/walberla/walberla>.

## Layout

```
apps/
├── CMakeLists.txt          # top-level driver: adds walberla once + each app
├── CMakePresets.json       # workstation / woody / LUMI configure + build presets
├── CMakeUserPresets.json
├── README.md               # this file
├── LICENSE                 # GPL-3.0-or-later (apps only)
├── walberla/               # git submodule (shared)
├── build/                  # CMake build output (gitignored)
├── HeatEquationKernel/     # heat-equation demo app
└── CouetteFlow/            # Couette flow benchmark app
```

A single CMake configure builds both apps. Each app is independently
switchable via `-DBUILD_HEAT_EQUATION=ON/OFF` and `-DBUILD_COUETTE_FLOW=ON/OFF`.

## Prerequisites

```bash
# From the repository root: get the source and the waLBerla submodule
git clone https://github.com/rahildoshi97/materforge.git
cd materforge
git submodule update --init --recursive      # populates apps/walberla

# Python toolchain for code generation
pip install -e .                              # MaterForge itself
pip install lbmpy pystencils pystencilssfg sweepgen

# Activate the virtualenv before any cmake/python invocation; the presets
# expect $env{VIRTUAL_ENV}/bin/python3.
source ~/.venvs/materforge/bin/activate       # adjust to your venv
```

You also need a C++20 compiler, CMake ≥ 3.24, and MPI. GPU builds additionally
need ROCm/HIP (LUMI-G) or CUDA.

## Build & run

```bash
cd apps/

# Configure (CPU release) and build both apps
cmake --preset local-release-cpu
cmake --build --preset local-release-cpu-build

# Binaries:
#   build/local-release-cpu/HeatEquationKernel/HeatEquationWithMaterial
#   build/local-release-cpu/CouetteFlowScaling
```

See [`HeatEquationKernel/README.md`](HeatEquationKernel/README.md) and
[`CouetteFlow/README.md`](CouetteFlow/README.md) for per-app configure options,
run instructions, and (for CouetteFlow) the SLURM sweep + post-processing
workflow.
