"""Exploratory annotation associations with the study's saved signed item-PC1 correlations.
Uses questionnaire-cluster bootstrap, not leave-one-questionnaire-out PC1.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
KEYS = ['inner_life_content','observable_behavior','beliefs_judgments',
        'human_world_dependence','valence','social_desirability']
a = pd.read_csv(OUT / 'item_means.csv')
p = pd.read_csv(OUT / 'item_pc1_correlations.csv')
def canonical_q(s):
    return s.strip().replace('Moral Foundations Questionnaire (MFQ-30)_1',
        'Moral Foundations Questionnaire (MFQ-30)').replace('Moral Foundations Questionnaire (MFQ-30)_2',
        'Moral Foundations Questionnaire (MFQ-30)')
for frame, text_column in [(a, 'item_text'), (p, 'item')]:
    frame['questionnaire_key'] = frame.questionnaire.map(canonical_q)
    frame['text_key'] = frame[text_column].str.strip()
join = ['questionnaire_key', 'text_key']
# Repeated wording elsewhere in the PC1 table must not create ambiguous matches.
relevant = p.merge(a[join], on=join, how='inner', validate='many_to_one')
m = a.merge(relevant[join+['r_pc1','source_item_indices','n_models']], on=join, how='left', validate='one_to_one')
m[m.r_pc1.isna()].to_csv(OUT / 'pc1_unmatched_items.csv', index=False)
d = m.dropna(subset=['r_pc1']).reset_index(drop=True)
d.to_csv(OUT / 'pc1_matched_items.csv', index=False)
# Verify reference sign using the exact values used in the manuscript's pole table.
assert p.loc[p.item.eq('Once I’ve made up my mind other people can seldom change my opinion.'),'r_pc1'].iloc[0] > 0
assert p.loc[p.item.eq('I often act on the spur of the moment.'),'r_pc1'].iloc[0] < 0
rng = np.random.default_rng(20260929)
groups = [g.index.to_numpy() for _,g in d.groupby('questionnaire_key')]
B = 5000
samples = [np.concatenate([groups[i] for i in draw]) for draw in rng.integers(0,len(groups),(B,len(groups)))]
results = []
for key in KEYS + ['inner_life_minus_behavior']:
    x,y = d[key].to_numpy(), d.r_pc1.to_numpy()
    r,pval = pearsonr(x,y)
    rho,sp = spearmanr(x,y)
    boots = np.array([(np.corrcoef(x[idx],y[idx])[0,1],spearmanr(x[idx],y[idx]).statistic) for idx in samples])
    cis = np.quantile(boots,[.025,.975],axis=0)
    complete = d[d.complete_three_rater_item]
    results.append(dict(rubric=key,n=len(d),n_questionnaires=len(groups),pearson_r=r,
        pearson_p_iid=pval,pearson_ci_low=cis[0,0],pearson_ci_high=cis[1,0],
        spearman_rho=rho,spearman_p_iid=sp,spearman_ci_low=cis[0,1],spearman_ci_high=cis[1,1],
        complete_n=len(complete),complete_pearson_r=pearsonr(complete[key],complete.r_pc1).statistic,
        complete_spearman_rho=spearmanr(complete[key],complete.r_pc1).statistic))
summary = pd.DataFrame(results)
summary.to_csv(OUT / 'pc1_correlations.csv',index=False)
# Exploratory joint regression: Fisher-z item-PC1 outcome, standardized predictors.
# Cluster-robust SEs account for dependence within questionnaires; full-PC1 outcome
# still contains each item's own questionnaire, and uncertain PC1 is held fixed.
regressions=[]
for label, subset in [('available',d),('complete',d[d.complete_three_rater_item])]:
    y = np.arctanh(subset.r_pc1)
    y = (y-y.mean())/y.std()
    for model,keys in [('primary',KEYS[:2]),('controls',KEYS[2:]),('full',KEYS)]:
        X=subset[keys];X=(X-X.mean())/X.std()
        fit=OLS(y,add_constant(X)).fit(cov_type='cluster',cov_kwds={'groups':subset.questionnaire_key},use_t=True)
        ci=fit.conf_int()
        for k in keys:
            regressions.append(dict(sample=label,model=model,predictor=k,n=len(subset),r_squared=fit.rsquared,
                standardized_beta=fit.params[k],ci_low=ci.loc[k,0],ci_high=ci.loc[k,1],p_cluster=fit.pvalues[k]))
reg=pd.DataFrame(regressions)
reg.to_csv(OUT / 'pc1_exploratory_regressions.csv',index=False)
meta=dict(bootstrap_replicates=B,seed=20260929,bootstrap_unit='questionnaire',
          matched_items=len(d),matched_questionnaires=len(groups),unmatched_items=int(m.r_pc1.isna().sum()),
          pc1_source='results/item_pc1_correlations.csv',
          pc1_source_sha256=hashlib.sha256((OUT/'item_pc1_correlations.csv').read_bytes()).hexdigest(),
          caveat='Full-precision item associations with the 45-questionnaire PC1; not leave-one-questionnaire-out; component and annotators held fixed; exploratory.')
(OUT/'pc1_analysis_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
print(summary.round(5).to_string(index=False))
print('\nRegressions:')
print(reg[reg['sample'].eq('available')].round(5).to_string(index=False))
