# Assess Fit Quality

A file-import, tabular, or computed property is built by fitting (regression) or
interpolating the data points it was given. MaterForge keeps those source points
on the material, so after a build you can ask **how faithfully the stored curve
reproduces its data** - without re-parsing the YAML or re-reading any data file.

All of this lives in `materforge.analysis`; the headline names are also re-exported
from the top-level `materforge` package.

---

## Score One Property

```python
import sympy as sp
import materforge as mf

T = sp.Symbol('T')
mat = mf.create_material('steel.yaml', dependency=T, enable_plotting=False)

fq = mf.fit_quality(mat, 'heat_capacity')
print(fq)
# heat_capacity: R²=0.999635  RMSE=8.939  MAE=4.214  max|err|=69.09  (n=541)

fq.rmse        # 8.939   - a plain float, in the property's units
fq.r_squared   # 0.999635
fq.n_points    # 541
```

`fit_quality` returns a `FitQuality` dataclass with `r_squared`, `rmse`, `mae`,
`max_abs_error`, and `n_points`. It evaluates the stored property at the source
sample points and compares the result with the sample values.

---

## Score Every Data-Backed Property

```python
report = mf.fit_report(mat)          # dict: property name -> FitQuality
for quality in report.values():
    print(quality)
```

`fit_report` compiles the material once and covers every property that retained
source data. Constants, step functions, and piecewise-equation properties are
exact definitions with no data to fit, so they are not included.

---

## Inspect Residuals

```python
x, res = mf.residuals(mat, 'heat_capacity')   # res = predicted - observed
worst = float(abs(res).max())
```

`residuals` returns two NumPy arrays - the dependency values and the per-point
errors - handy for a custom plot or a tolerance check in a test.

---

## What "Fit Quality" Means

The metric compares the **stored property** against the **data it was built from**:

- `regression: {simplify: pre}` or `{simplify: post}` - the stored curve *is* the
  regression, so R²/RMSE are the genuine fit error. This is the number you usually
  want.
- Plain interpolation (no `regression` block) - the stored curve passes through
  every point by construction, so the error reads ~0. Correct, if uninformative.

Fit quality is only available for **data-backed** properties (file-import,
tabular, computed). Asking for any other property raises `KeyError` with the list
of properties that do have source data.

---

## Metrics On Raw Arrays

The metric functions also work on plain array-likes, with no material involved -
useful in tests or when comparing your own predictions:

```python
from materforge import r_squared, rmse, mae, max_abs_error

r_squared([1, 2, 3], [1.1, 2.0, 2.9])   # 0.99
rmse(y_true, y_pred)
```

---

## A Note On Multiple Dependencies

Like `Material.compile`, the material-aware helpers evaluate against a single
dependency symbol, inferred from the properties. Pass `symbol=` to choose one
explicitly. Materials that depend on more than one symbol are a planned future
feature.

---

## Next Steps

- [Visualize properties](visualize_properties.md) - plot the fit, its residuals,
  and compare materials
- [API reference](../reference/api.rst)
