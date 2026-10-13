"""Variance ratios, loading changes, compression checks, and model-score convergence."""
from common import *
from scipy.stats import pearsonr,spearmanr
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

def corr(x,y):
    d=pd.DataFrame({'x':x,'y':y}).dropna()
    return dict(n=len(d),pearson_r=float(pearsonr(d.x,d.y).statistic),spearman_rho=float(spearmanr(d.x,d.y).statistic))

def loading_matrix():
    return pd.concat([loadings(c).assign(condition=c) for c in CONDITIONS]).pivot_table(index=['questionnaire','item'],columns='condition',values='loading').reset_index()

r=responses().dropna(subset=['response'])
rows=[]
for (q,idx),g in r.groupby(['questionnaire','item_index']):
    n=g.loc[g.condition=='neutral','response'];h=g.loc[g.condition=='human_simulation','response']
    if len(n)<5 or len(h)<5 or h.var()==0:continue
    rows.append(dict(questionnaire=q,item_index=idx,item=g['item'].iloc[0],var_neutral=round(float(n.var()),3),var_human_sim=round(float(h.var()),3),variance_ratio=round(float(n.var()/h.var()),3),n_neutral=len(n),n_human_sim=len(h)))
ratios=pd.DataFrame(rows).sort_values('variance_ratio',ascending=False)
assert len(ratios)==1312
ratios.to_csv(OUT/'variance_ratios.csv',index=False)
merged=loading_matrix().merge(ratios,on=['questionnaire','item'],how='inner')
merged['log_ratio']=np.log1p(merged.variance_ratio.clip(upper=merged.variance_ratio.quantile(.99)))
merged['delta']=merged.human_simulation-merged.neutral
pd.DataFrame([dict(outcome=c,**corr(merged.log_ratio,merged[c])) for c in CONDITIONS+['delta']]).to_csv(OUT/'loading_change_associations.csv',index=False)
merged.to_csv(OUT/'loading_change_items.csv',index=False)
# Three-condition variance comparison uses a model-by-item-index matrix.
wide={c:r[r.condition==c].pivot_table(index='model',columns=['questionnaire','item_index'],values='response') for c in CONDITIONS}
common=wide['neutral'].columns
for c in CONDITIONS[1:]:common=common.intersection(wide[c].columns)
models=sorted(set.intersection(*[set(w.index) for w in wide.values()]))
M={c:wide[c].loc[models,common].to_numpy(float) for c in CONDITIONS}
var={c:np.nanvar(m,axis=0,ddof=1) for c,m in M.items()}
keep=np.ones(len(common),dtype=bool)
for c in CONDITIONS:keep &= (np.isfinite(M[c]).sum(axis=0)>=5)&(var[c]>0)
text=r.drop_duplicates(['questionnaire','item_index']).set_index(['questionnaire','item_index'])['item']
items=pd.DataFrame({'questionnaire':[q for q,i in common],'item_index':[i for q,i in common],'item':[text.loc[key] for key in common],**{c:var[c] for c in CONDITIONS}})[keep].reset_index(drop=True)
assert len(items)==1310
M={c:m[:,keep] for c,m in M.items()}
gm={c:float(np.exp(np.log(items[c]).mean())) for c in CONDITIONS}
save_json('variance_compression.json',dict(n_items=len(items),geometric_mean_variances=gm,human_to_neutral=gm['human_simulation']/gm['neutral'],analog_to_neutral=gm['llm_analog']/gm['neutral']))
# Fixed supplied partitions index the alphabetically sorted 50-model panel.
partitions=pd.read_csv(DATA/'model_splits.csv').to_numpy()
assert partitions.shape==(500,50) and all(np.array_equal(np.sort(p),np.arange(50)) for p in partitions)
rhos=[]
for perm in partitions:
    a,b=perm[:25],perm[25:]
    with np.errstate(divide='ignore',invalid='ignore'):
        ra=np.log(np.nanvar(M['neutral'][a],axis=0,ddof=1)/np.nanvar(M['human_simulation'][a],axis=0,ddof=1))
        rb=np.log(np.nanvar(M['neutral'][b],axis=0,ddof=1)/np.nanvar(M['human_simulation'][b],axis=0,ddof=1))
    ok=np.isfinite(ra)&np.isfinite(rb)
    rhos.append(spearmanr(ra[ok],rb[ok]).statistic)
pd.DataFrame({'split':np.arange(1,501),'spearman_rho':rhos}).to_csv(OUT/'split_half_correlations.csv',index=False)
save_json('split_half_summary.json',dict(n_splits=500,mean=float(np.mean(rhos)),lower=float(np.quantile(rhos,.025)),upper=float(np.quantile(rhos,.975))))
items['log_ratio']=np.log(items.neutral/items.human_simulation)
items['within_questionnaire']=items.log_ratio-items.groupby('questionnaire').log_ratio.transform('mean')
items['variance_residual']=-OLS(np.log(items.human_simulation),add_constant(np.log(items.neutral))).fit().resid
# Loading changes use the absolute first loading in each condition.
weights=loading_matrix().rename(columns={c:'loading_'+c for c in CONDITIONS})
j=items.merge(weights,on=['questionnaire','item'],how='inner')
j['delta']=j.loading_human_simulation-j.loading_neutral
variants=['log_ratio','within_questionnaire','variance_residual']
pd.DataFrame([dict(measure=v,**corr(j[v],j.delta)) for v in variants]).to_csv(OUT/'alternative_ratio_associations.csv',index=False)
j.to_csv(OUT/'alternative_ratio_items.csv',index=False)
# Variance-ratio-weighted model score, positive capped log weights.
n=r[r.condition=='neutral'].pivot_table(index='model',columns=['questionnaire','item'],values='response')
z=(n-n.mean())/n.std(ddof=0)
w=ratios[ratios.variance_ratio>1].copy();w['weight']=np.log(w.variance_ratio.clip(upper=ratios.variance_ratio.quantile(.99)))
w=w[w.weight>0].drop_duplicates(['questionnaire','item'],keep='last').set_index(['questionnaire','item']).weight
cols=z.columns.intersection(w.index);z=z[cols];w=w.reindex(cols)
score=z.mul(w).sum(axis=1)/z.notna().mul(w).sum(axis=1)
pc=pd.read_csv(OUT/'global_pca_model_scores.csv').set_index('model').PC1
joined=pd.concat([pc,score.rename('ratio_weighted_score')],axis=1).dropna()
joined.to_csv(OUT/'variance_weighted_model_scores.csv')
save_json('variance_weighted_convergence.json',corr(joined.PC1,joined.ratio_weighted_score))
print('Framing checks complete:',len(ratios),'ratios;',len(items),'three-condition items')

# Apply the same neutral/human-simulation weights to analog responses.
a=r[r.condition=='llm_analog'].pivot_table(index='model',columns=['questionnaire','item'],values='response')
a=(a-a.mean())/a.std(ddof=0)
ac=a.columns.intersection(w.index);a=a[ac];aw=w.reindex(ac)
analog=a.mul(aw).sum(axis=1)/a.notna().mul(aw).sum(axis=1)
comparison=pd.concat([score.rename('neutral'),analog.rename('llm_analog')],axis=1).dropna()
comparison.to_csv(OUT/'weighted_condition_scores.csv')
save_json('weighted_condition_comparison.json',corr(comparison.neutral,comparison.llm_analog))
