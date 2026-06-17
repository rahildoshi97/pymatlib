# Evaluate Properties Quickly

`Material.evaluate(symbol, value)` substitutes the dependency symbolically and
returns a new `Material`. That is convenient for a one-off value, but each call
re-does the symbolic work, and it only accepts a single scalar. When you need to
evaluate over many dependency values - sweeping a temperature range, sampling a
property for a plot or an export - compile the material once and reuse it.

`Material.compile()` returns a `MaterialEvaluator` that lambdifies every property
into a fast NumPy callable and caches it. Calling the evaluator with an array
evaluates all properties over the whole array in one vectorised pass.

---

## Compile Once, Evaluate Many Times

```python
import numpy as np
import sympy as sp
from materforge import create_material

T = sp.Symbol('T')
mat = create_material('steel.yaml', dependency=T, enable_plotting=False)

evaluate = mat.compile()        # lambdifies + caches every property

# Scalar in -> dict of floats
evaluate(500.0)
# {'density': 7861.2, 'heat_capacity': 488.1, ...}

# Array in -> dict of NumPy arrays, one call per property
temperatures = np.linspace(300, 1800, 500)
table = evaluate(temperatures)
table['density'].shape        # (500,)
```

A scalar input gives plain Python floats; an array input gives NumPy arrays
shaped like the input. Constant properties (those that do not depend on the
symbol) are broadcast to match the array.

---

## Why It Is Faster

`compile()` does the expensive symbolic-to-numeric conversion (`sympy.lambdify`)
**once per property** and stores the resulting callable. Repeated evaluations -
and array evaluations - then run as compiled NumPy, skipping the per-call
symbolic substitution that `evaluate()` performs every time.

The evaluator is a **snapshot** taken when you call `compile()`. If you change a
material's properties afterwards, build a fresh evaluator.

---

## Choosing the Dependency Symbol

By default the symbol is inferred from the properties - the common case, where
every property is a function of one dependency (the symbol you passed to
`create_material`). Pass `symbol=` to be explicit, for instance with a material
whose properties are all constants:

```python
mat.compile(symbol=sp.Symbol('u_C'))
```

A material whose properties depend on more than one symbol raises `ValueError`;
multivariate evaluation is a planned future feature.

---

## Getting a Single Property's Callable

When you only need one property - or want to hand the raw function to another
numerical routine - ask for its callable directly:

```python
density = mat.compile().function('density')
density(np.linspace(300, 1800, 100))   # NumPy array
```

---

## When to Use Which

- **`material.evaluate(symbol, value)`** - a single scalar, when you want a
  `Material` back (e.g. to read `result.density` by attribute).
- **`material.compile()`** - many values or an array, when you want raw numbers
  fast.

---

## Next Steps

- [Use the command-line interface](use_the_cli.md) - `materforge evaluate` for a
  one-off value from the shell
- [API reference](../reference/api.rst)
