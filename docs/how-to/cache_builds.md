# Reuse Builds With the Cache

Building a material runs a piecewise `pwlf` regression for every tabular or
file-import property. That fit is the slowest part of `create_material`, and it
produces the same result every time the source is unchanged. MaterForge caches
the built result on disk so a second load of an unchanged YAML returns instantly,
skipping the regression entirely.

The cache is on by default for non-plotting builds and needs no setup.

---

## How It Works

```python
import sympy as sp
from materforge import create_material

T = sp.Symbol('T')

# First call: builds normally and stores the result.
mat = create_material('steel.yaml', dependency=T, enable_plotting=False)

# Later call (same process or a new one): returns the cached build,
# skipping the pwlf regression.
mat = create_material('steel.yaml', dependency=T, enable_plotting=False)
```

A cache entry is keyed by a hash of everything that can change the result:

- the YAML file contents,
- the contents of every data file it references via `file_path`,
- the dependency symbol name, and
- the installed `materforge` and `sympy` versions.

Edit the YAML or any referenced data file, pass a different dependency symbol, or
upgrade either library, and the key changes - so you get a fresh build
automatically. There is no stale-cache trap: an entry is reused only when it
provably matches.

---

## When the Cache Is Used

The cache is consulted only when `enable_plotting=False`. A plotting run exists to
(re)produce figures, so it always rebuilds. Opt out for a single call with
`use_cache=False`:

```python
create_material('steel.yaml', dependency=T, enable_plotting=False, use_cache=False)
```

---

## Where It Lives, and Turning It Off

| What | How |
|------|-----|
| Default location | `~/.cache/materforge/` (honours `XDG_CACHE_HOME`) |
| Custom location | set `MATERFORGE_CACHE_DIR=/path/to/dir` |
| Disable entirely | set `MATERFORGE_DISABLE_CACHE=1` |
| Clear it | `from materforge import clear_cache; clear_cache()` |

The cache is always safe to delete by hand: any missing, unreadable, or
version-mismatched entry is silently ignored and rebuilt.

```python
from materforge import clear_cache

removed = clear_cache()   # returns the number of entries deleted
```

---

## Next Steps

- [Evaluate properties quickly](fast_evaluation.md) - compile a built material
  for fast repeated numeric evaluation
- [API reference](../reference/api.rst)
