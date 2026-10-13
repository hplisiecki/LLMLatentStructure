"""Full-precision item associations with the shared model component."""
from common import *
r=responses();r=r[(r.condition=='neutral')&r.response.notna()]
pc=pd.read_csv(OUT/'global_pca_model_scores.csv').set_index('model').PC1
r=r[r.model.isin(pc.index)]
rows=[];excluded=[]
for (q,item),group in r.groupby(['questionnaire','item']):
    v=group.groupby('model').response.mean()
    base=dict(questionnaire=q,item=item,n_models=len(v),source_item_indices='|'.join(map(str,sorted(group.item_index.unique()))))
    reason='insufficient_coverage' if len(v)<15 else ('constant' if v.std()==0 else None)
    if reason:excluded.append(dict(**base,reason=reason))
    else:rows.append(dict(**base,r_pc1=v.corr(pc)))
pd.DataFrame(rows).sort_values('r_pc1',ascending=False).to_csv(OUT/'item_pc1_correlations.csv',index=False)
pd.DataFrame(excluded).to_csv(OUT/'item_correlation_exclusions.csv',index=False)
assert len(rows)==1325
print('Item correlations:',len(rows))
