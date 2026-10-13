"""Item means, ordinal agreement, and annotator category-use diagnostics."""
from common import *
import krippendorff
raw=pd.read_csv(DATA/'annotations/ratings.csv')
eligible=pd.read_csv(DATA/'annotations/eligible_items.csv')
assert not raw.duplicated(['item_id','annotator_id']).any()
assert raw.groupby('item_id').annotator_id.nunique().eq(3).all()
means = raw.groupby('item_id')[RUBRICS].mean().reset_index()
means = means.merge(eligible, on='item_id', validate='one_to_one')
means['n_annotators'] = 3
for key in RUBRICS:
    counts = raw.groupby('item_id')[key].count()
    means[key + '_n'] = means.item_id.map(counts)
means['complete_three_rater_item'] = means[[k + '_n' for k in RUBRICS]].eq(3).all(axis=1)
means['inner_life_minus_behavior'] = means.inner_life_content - means.observable_behavior
means.to_csv(OUT / 'item_means.csv', index=False)
means.loc[means.complete_three_rater_item].to_csv(OUT / 'item_means_complete.csv', index=False)

rng = np.random.default_rng(20260929)
boot_indices = rng.integers(0, len(means), (2000, len(means)))
reliability, by_rater, pairs = [], [], []
for key in RUBRICS:
    mat = raw.pivot(index='item_id', columns='annotator_id', values=key)
    a = mat.to_numpy().T
    alpha = krippendorff.alpha(reliability_data=a, level_of_measurement='ordinal', value_domain=np.arange(1,8))
    boots = [krippendorff.alpha(reliability_data=a[:, idx], level_of_measurement='ordinal', value_domain=np.arange(1,8)) for idx in boot_indices]
    lo, hi = np.quantile(boots, [.025, .975])
    reliability.append(dict(rubric=key, mean=raw[key].mean(), ordinal_alpha=alpha, ci_low=lo, ci_high=hi))
    for who, sub in raw.groupby('annotator_id'):
        rec = dict(rubric=key, annotator_id=who, n_records=len(sub), n_valid=sub[key].count(), mean=sub[key].mean(), sd=sub[key].std(),
                   floor_pct=100*sub[key].dropna().eq(1).mean(), ceiling_pct=100*sub[key].dropna().eq(7).mean())
        rec.update({f'n_{v}': int(sub[key].eq(v).sum()) for v in range(1,8)})
        by_rater.append(rec)
    for i in range(3):
        for j in range(i+1,3):
            valid = np.isfinite(a[i]) & np.isfinite(a[j])
            diff = np.abs(a[i, valid]-a[j, valid])
            pairs.append(dict(rubric=key, rater_a=mat.columns[i], rater_b=mat.columns[j], n=len(diff),
                              exact_agreement_pct=100*np.mean(diff==0), within_one_pct=100*np.mean(diff<=1),
                              mean_absolute_difference=np.mean(diff)))
rel = pd.DataFrame(reliability)
rel.to_csv(OUT / 'reliability.csv', index=False)
pd.DataFrame(by_rater).to_csv(OUT / 'rater_summary.csv', index=False)
pd.DataFrame(pairs).to_csv(OUT / 'pairwise_agreement.csv', index=False)
means[RUBRICS].corr(method='spearman').to_csv(OUT / 'rubric_correlations.csv')

print('Rated items:',len(means),'observed cells:',raw[RUBRICS].count().sum())
