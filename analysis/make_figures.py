"""Create manuscript figures; bootstrap fixed questionnaire summaries, seed 42."""
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/figures'
OUT.mkdir(parents=True,exist_ok=True)
DATA = ROOT / 'results'
# Reuse the manuscript's scoring and reference orientation exactly.
d = runpy.run_path(str(ROOT / 'analysis/global_structure.py'))
x, models, reference = d['x'], d['s'].index, d['pc'][:, 0]
rng = np.random.default_rng(42)
boot = []
with threadpool_limits(limits=1):
    for _ in range(1000):
        # Integer indexing retains every repeated questionnaire draw.
        selected = rng.integers(0, x.shape[1], x.shape[1])
        xb = x[:, selected]
        assert xb.shape == (50, 45)
        _, vectors = np.linalg.eigh(xb.T @ xb)
        score = xb @ vectors[:, -1]
        if np.dot(score, reference) < 0:
            score = -score
        score = score / score.std() * reference.std()
        boot.append(score)
boot = np.asarray(boot)
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
result = pd.DataFrame({'model': models, 'PC1': reference, 'lower': lo, 'upper': hi})
result.to_csv(DATA / 'global_pca_model_bootstrap.csv', index=False)
assert np.isfinite(boot).all()
assert np.allclose(np.std(boot, axis=1), reference.std())
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
blue = '#246a88'
provider_counts = result.model.str.split('/').str[0].value_counts()
provider_styles = {
    'openai': ('OpenAI', '#009E73', '*'),
    'anthropic': ('Anthropic', '#D55E00', 's'),
    'google': ('Google', '#0072B2', '^'),
    'meta-llama': ('Meta', '#56B4E9', 'D'),
    'cohere': ('Cohere', '#CC79A7', 'v'),
    'x-ai': ('xAI', '#E69F00', 'P'),
    'qwen': ('Qwen', '#6F4C9B', 'X'),
}
highlighted = {p: style for p, style in provider_styles.items()
               if provider_counts.get(p, 0) >= 4}
assert set(highlighted) == set(provider_counts[provider_counts >= 4].index)
other_color = '#858585'
fig = plt.figure(figsize=(6.3, 8.4))
# Small spectrum panel leaves enough height for legible model names.
ax = fig.add_axes([.36, .875, .60, .09])
e = d['e'][:10]
ax.bar(np.arange(1, 11), e, color=[blue] + ['#b6c9d0'] * 9, width=.7)
ax.set(xticks=np.arange(1, 11), ylim=(0, 57), yticks=[0, 25, 50])
ax.set_ylabel('Variance (%)', fontsize=9.5)
ax.set_xlabel('Global principal component', fontsize=9.5, labelpad=2)
for i in range(3):
    ax.text(i+1, e[i] + (8 if i == 1 else 1), f'{e[i]:.1f}%', ha='center', fontsize=9)
fig.text(.025, .972, 'a  Shared structure', weight='bold', fontsize=9)
fig.text(.025, .81, 'b  Model scores', weight='bold', fontsize=9)
ranked = result.sort_values('PC1', ascending=False)
legend_markers = {}
for part, left in enumerate([.255, .765]):
    ax = fig.add_axes([left, .12, .215, .66])
    half = 1 - part  # Lower scores on the left; higher scores on the right.
    sub = ranked.iloc[half*25:(half+1)*25]
    y = np.arange(25)
    ax.axvline(0, color='#8b959b', lw=.7, linestyle='--')
    ax.hlines(y, sub.lower, sub.upper, color='#bcbcbc', lw=1.5)
    for row, (_, model_row) in enumerate(sub.iterrows()):
        label, color, marker = highlighted.get(
            model_row.model.split('/', 1)[0], ('Other providers', other_color, 'o'))
        artist = ax.scatter(model_row.PC1, row, s=44 if marker == '*' else 26, color=color, marker=marker,
                   edgecolors='white', linewidths=.25, zorder=3, label=label)
        legend_markers.setdefault(label, artist)
    labels = [m.split('/', 1)[-1] for m in sub.model]
    labels = [label[:cut + 1] + '\n' + label[cut + 1:]
              if len(label) > 22 else label
              for label in labels for cut in [label.rfind('-', 0, 21)]]
    ax.set(yticks=y, yticklabels=labels, ylim=(24.7, -.7), xlim=(-13, 11),
           xticks=[-10, 0, 10], xlabel='Global PC1 score')
    ax.tick_params(axis='y', length=0, labelsize=9, pad=3)
    for label in ax.get_yticklabels():
        label.set_linespacing(.85)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', color='#e6eaed', lw=.5)
    ax.set_axisbelow(True)
# Reuse the plotted scatter artists so legend paths, sizes, and edges match.
legend_labels = [style[0] for style in highlighted.values()] + ['Other providers']
handles = [legend_markers[label] for label in legend_labels]
fig.legend(handles=handles, labels=legend_labels, markerscale=1, scatterpoints=1,
           loc='lower center', bbox_to_anchor=(.5, .005),
           ncol=4, frameon=False, fontsize=9, columnspacing=1.6,
           handletextpad=.4, labelspacing=.55)
fig.savefig(OUT / 'global_pca_models.pdf')
fig.savefig(OUT / 'global_pca_models.png', dpi=200)
plt.close(fig)

r = pd.read_csv(DATA / 'pc1_exploratory_regressions.csv')
r = r[(r['sample']=='available') & (r['model']=='full')]
assert len(r)==6 and r.r_squared.nunique()==1
labels = ['Inner-life content', 'Observable behavior', 'Beliefs/judgments',
          'Human-world\ndependence', 'Valence', 'Social desirability']
fig, ax = plt.subplots(figsize=(3.05, 3.25))
fig.subplots_adjust(left=.53, right=.97, bottom=.25, top=.84)
y = np.arange(6)
ax.axvline(0, color='#8b959b', lw=.8, linestyle='--')
ax.hlines(y, r.ci_low, r.ci_high, color='#9bb9c6', lw=2)
ax.scatter(r.standardized_beta, y, s=28, color=blue, zorder=3)
ax.set(yticks=y, yticklabels=labels, ylim=(5.65, -.65), xlim=(-.72, .48),
       xticks=[-.6, -.3, 0, .3], xlabel='Standardized\ncoefficient ($\\beta$)')
ax.tick_params(axis='y', length=0, labelsize=9.5, pad=4)
ax.tick_params(axis='x', labelsize=9.5)
ax.xaxis.label.set_size(9.5)
ax.spines['left'].set_visible(False)
ax.grid(axis='x', color='#e6eaed', lw=.5)
ax.set_axisbelow(True)
fig.text(.5, .94, rf'Joint model: $R^2 = {r.r_squared.iloc[0]:.3f}$', fontsize=9.5, ha='center')
fig.savefig(OUT / 'annotation_coefficients.pdf')
fig.savefig(OUT / 'annotation_coefficients.png', dpi=220)
plt.close(fig)
print('Saved both manuscript figures and 1,000-draw model bootstrap.')
