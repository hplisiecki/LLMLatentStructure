"""Derive neutral extraction diagnostics without refitting the study's factors."""
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
r = pd.read_csv(ROOT / 'data/responses.csv')
r['response'] = pd.to_numeric(r['response'], errors='coerce')
r = r[(r.condition == 'neutral') & r.response.notna()]
summary = pd.read_csv(ROOT / 'data/factor_loadings/summary.csv')
rows = []
for row in summary[summary.condition == 'neutral'].itertuples():
    q = row.questionnaire
    safe = re.sub(r'[\\/*?:"<>|]', '_', q)
    loading = pd.read_csv(ROOT / f'data/factor_loadings/{safe}__neutral.csv', index_col=0)
    matrix = r[r.questionnaire == q].pivot_table(index='model', columns='item', values='response').dropna()
    matrix = matrix.loc[:, matrix.std() > 0]
    assert matrix.shape == (row.n_models, row.n_items), q
    assert set(matrix.columns) == set(loading.index), q
    matrix = matrix.loc[:, loading.index]
    first = loading.iloc[:, 0].to_numpy()
    if row.method == 'PCA':
        centered = matrix.to_numpy() - matrix.to_numpy().mean(axis=0)
        fraction = np.var(centered @ first, ddof=1) / np.var(centered, axis=0, ddof=1).sum()
        singular = np.linalg.svd(centered, compute_uv=False)
        assert np.isclose(fraction, singular[0] ** 2 / np.square(singular).sum(), atol=1e-6), q
        retained = np.var(centered @ loading.to_numpy(), axis=0, ddof=1)
        first_share = retained[0] / retained.sum()
        assert np.isclose(first_share, singular[0] ** 2 / np.square(singular[:row.n_factors]).sum(), atol=1e-6), q
        metric = 'PC1 share of total unstandardized item variance'
    else:
        fraction = np.square(first).sum() / row.n_items
        retained = np.square(loading.to_numpy()).sum(axis=0)
        first_share = retained[0] / retained.sum()
        metric = 'F1 sum of squared pattern loadings divided by item count'
    rows.append(dict(questionnaire=q, method=row.method, n_models=row.n_models,
                     n_items=row.n_items, n_factors=row.n_factors,
                     first_dimension_percent=100 * fraction,
                     first_share_of_retained_percent=100 * first_share, metric=metric))
result = pd.DataFrame(rows)
result.to_csv(ROOT / 'results/neutral_extraction_variance.csv', index=False)
print(result[['questionnaire','method','first_dimension_percent']].to_string(index=False))
