"""Verify analysis populations and statistics quoted in the manuscript."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parents[1];out=root/'results';checks=[]
def csv(name):return pd.read_csv(out/(name+'.csv'))
def js(name):return json.loads((out/(name+'.json')).read_text())
def check(label,actual,expected,tolerance=1e-7):
    assert np.isclose(actual,expected,atol=tolerance,rtol=0),(label,actual,expected)
    checks.append(dict(check=label,actual=float(actual),expected=float(expected),absolute_tolerance=tolerance))
check('global model count',len(csv('global_pca_model_scores')),50)
check('global questionnaire count',len(csv('global_pca_questionnaire_correlations')),45)
check('neutral PC1 variance percent',csv('global_pca_spectrum').variance_percent.iloc[0],47.13177136416)
check('analog PC1 variance percent',js('condition_comparison')['llm_analog_pc1_variance_percent'],41.34175511348)
check('eligible item associations',len(csv('item_pc1_correlations')),1325)
f=js('ssd_fit_summary')
for k,e in [('input_items',1325),('retained_items',1324),('K',16),('r',.26210425308),('r2',.06869863948)]:check('SSD '+k,f[k],e,1e-6)
check('ratio count',len(csv('variance_ratios')),1312)
a=csv('loading_change_associations').set_index('outcome').loc['delta']
check('primary loading-change N',a.n,1292)
check('primary loading-change rho',a.spearman_rho,-.2146676778)
check('primary loading-change r',a.pearson_r,-.1490354107)
a=csv('alternative_ratio_associations').set_index('measure')
for k,e in [('log_ratio',-.2147497134),('within_questionnaire',-.2712187138),('variance_residual',-.1910486320)]:
    check(k+' loading-change rho',a.loc[k,'spearman_rho'],e)
    check(k+' loading-change N',a.loc[k,'n'],1291)
check('split-half mean',js('split_half_summary')['mean'],.5747332075)
check('human/neutral geometric variance',js('variance_compression')['human_to_neutral'],.3939842959)
check('analog/neutral geometric variance',js('variance_compression')['analog_to_neutral'],1.2032783973)
check('weighted score vs PC1 r',js('variance_weighted_convergence')['pearson_r'],.8637338389)
check('weighted neutral/analog rho',js('weighted_condition_comparison')['spearman_rho'],.4816806723)
a=csv('repeated_response_summary').set_index('tier')
check('exact-prompt observed pairs',a.loc['A','n_observed_pairs'],139)
check('exact-prompt agreement',a.loc['A','exact_agreement'],.7050359712)
check('exact-prompt standardized correlation',a.loc['A','pooled_standardized_r'],.5502098512)
check('same-scale aggregate correlation',a.loc['B','aggregate_r'],.9154343681)
a=csv('pc1_correlations').set_index('rubric')
check('annotation matched N',a.loc['inner_life_content','n'],289)
check('annotation complete N',a.loc['inner_life_content','complete_n'],281)
check('inner-life item r',a.loc['inner_life_content','pearson_r'],.2245502114)
check('observable-behavior item r',a.loc['observable_behavior','pearson_r'],-.2265553228)
a=csv('pc1_exploratory_regressions');a=a[(a['sample']=='available')&(a.model=='full')]
check('six-dimension R squared',a.r_squared.iloc[0],.221334,1e-6)
a=csv('annotation_icc').set_index('rubric')
check('inner-life ICC3,3',a.loc['inner_life_content','icc_3_3'],.8750190629)
report=dict(status='passed',checks=checks,scope='Checks saved outputs against reported populations and numerical results; run_analysis.py regenerates outputs. SSD is rerun only when --embeddings is supplied.')
(root/'validation/result_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'{len(checks)} reported-result checks passed.')
