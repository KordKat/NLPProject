---
name: scientific-plotting
title: Publication-quality scientific plots with matplotlib
description: Produce clear, correctly labelled scientific figures (function comparisons, experimental data, distributions, error bars, multi-panel figures) with run_python and matplotlib.
triggers: plot, graph of, line graph, chart, figure, visualize, visualise, visualization, histogram, scatter, heatmap, curve, compare functions
tools: run_python, eval
---

# Scientific plotting

Use **run_python**; save every figure under `figures/` with a descriptive name. Saved figures are
attached to the answer automatically — refer to them by file name.

## Checklist for every figure
1. Decide the message first ("A grows faster than B after x≈2") and choose the plot type that shows it:
   lines for functions/time series, scatter for relationships, histogram/KDE for distributions,
   bar + error bars for group comparisons, heatmap for matrices, log axes for data spanning decades.
2. Label axes **with units**, add a legend when there is more than one series, and a short title.
3. Use enough samples for smooth curves (`np.linspace(a, b, 500)`); guard singularities (mask or `np.where`).
4. Use colour-blind-safe colours (default `tab10` is fine) and distinguish series by line style too.
5. Print the key numbers you will mention (maxima, crossings, errors) so the answer is grounded in computed values.

## Template
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-4, 4, 600)
fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
ax.plot(x, np.sin(x), label=r"$\sin x$", lw=2)
ax.plot(x, x - x**3/6, "--", label=r"$x - x^3/6$", lw=2)
ax.set(xlabel="x (rad)", ylabel="value", title="sin(x) vs cubic Taylor polynomial", ylim=(-2, 2))
ax.axhline(0, color="0.6", lw=0.8); ax.grid(alpha=0.3); ax.legend(frameon=False)
fig.savefig("figures/sin_vs_taylor.png", dpi=200)
err = np.abs(np.sin(x) - (x - x**3/6))
print("max |error| on [-4,4]:", err.max(), "at x =", x[err.argmax()])
```

## Variants
- Multi-panel: `fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)`; label panels (a), (b).
- Error bars: `ax.errorbar(x, mean, yerr=std, fmt="o", capsize=3)`; state what the bars mean (SD, SE, 95% CI).
- Log scale: `ax.set_yscale("log")`; never plot non-positive values on a log axis.
- Equations in labels use mathtext: `r"$\alpha_t$"` (no `\begin{...}` environments).

After the run, describe what the figure shows and the computed numbers; do not describe features you did not compute.
