---
name: data-analysis
title: Data analysis of uploaded datasets
description: Load, clean, summarise and statistically analyse tabular data files (CSV, TSV, JSON, Excel) uploaded to the project with pandas/scipy in run_python, and report results with figures.
triggers: data, dataset, csv, tsv, excel, xlsx, spreadsheet, table, statistics, statistical test, regression, correlation, t-test, anova, mean, median, analyze data, analyse data
tools: list_files, read_file, run_python
---

# Data analysis

## Workflow
1. **Locate the data:** list_files (uploads usually live in `uploads/`). Use the exact path returned.
2. **Inspect before analysing** (first run_python call): shape, dtypes, head, missing values, value ranges.
   ```python
   import pandas as pd
   df = pd.read_csv("uploads/data.csv")          # read_excel / read_json for other formats
   print(df.shape); print(df.dtypes); print(df.head()); print(df.isna().sum()); print(df.describe(include="all").T)
   ```
3. **Clean explicitly** and report what you changed (dropped rows, parsed dates, unit conversions). Never silently drop data.
4. **Analyse** to answer the user's actual question:
   - group comparisons → `df.groupby(...).agg(["mean", "std", "count"])`, then `scipy.stats.ttest_ind` / `mannwhitneyu` / `f_oneway`;
   - relationships → scatter plot + `scipy.stats.pearsonr` / `spearmanr`; linear fit with `np.polyfit` or `scipy.stats.linregress`;
   - distributions → histogram, normality check (`scipy.stats.shapiro`) when it matters.
5. **Visualise** key results (save under `figures/`), and save derived tables with `df.to_csv("results/<name>.csv", index=False)`.

## Reporting
- Report effect sizes and uncertainty (mean ± SD, 95% CI, p-values with the test name and n).
- Distinguish correlation from causation; mention confounders and data limitations.
- Quote only numbers that the code printed. If the code failed, read stderr, fix it and re-run once.
