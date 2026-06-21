# Visualize Properties

`create_material(..., enable_plotting=True)` writes a composite PNG of every
property while it builds. That is great for a one-shot overview, but it always
saves to disk and you cannot restyle it. For interactive work - a Jupyter
notebook, a custom figure, comparing materials - use the post-build helpers in
`materforge.visualization.plots` (re-exported from the top-level `materforge`).

Each helper draws onto a Matplotlib `Axes` (a fresh one, or the `ax=` you pass)
and **returns it without saving or closing** - so a notebook cell renders it
inline and you keep full control of styling and output.

---

## Plot a Property With Its Data

```python
import sympy as sp
import materforge as mf

T = sp.Symbol('T')
mat = mf.create_material('steel.yaml', dependency=T, enable_plotting=False)

ax = mf.plot_property(mat, 'heat_capacity')   # fitted curve + the source points
```

The curve is sampled from the compiled property over the source-data range;
`show_data=False` hides the scatter, and `dep_range=(lower, upper)` overrides the
range (required for a property that kept no source data, such as a constant or a
piecewise-equation).

---

## Plot Residuals

```python
ax = mf.plot_residuals(mat, 'heat_capacity')  # predicted - observed, with a zero line
```

A flat band near zero means a good fit; structure (a trend or fanning) points to
too few segments or too low a regression degree.

---

## Compare Materials

```python
ax = mf.compare_materials([steel, alloy], 'density', labels=['1.4301', 'myAlloy'])
```

Overlays the same property from several materials on one axis - useful for
comparing two materials, or the same material built with different regression
settings. `labels` default to each material's name.

---

## Compose Your Own Figure

Because every helper takes `ax=` and returns it, you can lay out exactly what you
want and decide if and where to save:

```python
import matplotlib.pyplot as plt

fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
mf.plot_property(mat, 'heat_capacity', ax=top)
mf.plot_residuals(mat, 'heat_capacity', ax=bottom)
fig.savefig('heat_capacity_fit.png', dpi=300)   # only if you want to
```

---

## Next Steps

- [Assess fit quality](assess_fit_quality.md) - the R²/RMSE numbers behind the plots
- [Evaluate properties quickly](fast_evaluation.md) - the compiled evaluator the
  plots are built on
- [API reference](../reference/api.rst)
