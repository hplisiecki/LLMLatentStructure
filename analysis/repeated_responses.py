"""Repeated-prompt diagnostics reported in Appendix D."""
from common import *
import re
from scipy.stats import pearsonr
r=responses(selected=False)
q=pd.read_csv(DATA/'questionnaire_registry.csv');q['quest']=q.quest.str.strip()
def bounds(s):
    v=[int(x) for x in re.findall(r'(\d+)\s*[=:]',str(s))]
    return (min(v),max(v)) if len(v)>1 else (np.nan,np.nan)
b={row.quest:bounds(row.scale_prompt) for row in q.itertuples()}
name='Moral Foundations Questionnaire (MFQ-30)';b[name]=b[name+'_1']
def norm(s):
    s=str(s).lower().replace('?',"'").replace('?',"'")
    return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]',' ',s)).strip()
slots=r.drop_duplicates(['questionnaire','item_index'])[['questionnaire','item_index','item']].copy()
slots['norm']=slots['item'].map(norm)
pairs=[]
for _,g in slots.groupby('norm'):
    records=g.to_dict('records')
    for i,u in enumerate(records):
        for v in records[i+1:]:
            same=u['questionnaire']==v['questionnaire'];scale=b.get(u['questionnaire'])==b.get(v['questionnaire'])
            pairs.append(dict(tier='A' if same else ('B' if scale else 'C'),q_a=u['questionnaire'],i_a=u['item_index'],q_b=v['questionnaire'],i_b=v['item_index'],item=u['item']))
pairs=pd.DataFrame(pairs);pairs.to_csv(OUT/'repeated_item_pairs.csv',index=False)
models=sorted(r.model.unique());neutral=r[r.condition=='neutral'].pivot_table(index='model',columns=['questionnaire','item_index'],values='response').reindex(models)
summary=[]
for tier in ['A','B']:
    observed=[];za=[];zb=[];A=[];B=[]
    for row in pairs[pairs.tier==tier].itertuples():
        av=neutral[(row.q_a,row.i_a)].to_numpy();bv=neutral[(row.q_b,row.i_b)].to_numpy();valid=np.isfinite(av)&np.isfinite(bv)
        if valid.sum()<10:continue
        observed.extend((av[valid]==bv[valid]).tolist())
        aa=np.full(len(models),np.nan);bb=aa.copy()
        if av[valid].std()>0 and bv[valid].std()>0:
            aa[valid]=(av[valid]-av[valid].mean())/av[valid].std();bb[valid]=(bv[valid]-bv[valid].mean())/bv[valid].std()
            za.extend(aa[valid]);zb.extend(bb[valid])
        A.append(aa);B.append(bb)
    ma=np.nanmean(A,axis=0);mb=np.nanmean(B,axis=0);ok=np.isfinite(ma)&np.isfinite(mb)
    summary.append(dict(tier=tier,n_items=len(A),n_observed_pairs=len(observed),exact_agreement=float(np.mean(observed)),pooled_standardized_r=float(pearsonr(za,zb).statistic),aggregate_r=float(pearsonr(ma[ok],mb[ok]).statistic),n_models=int(ok.sum())))
pd.DataFrame(summary).to_csv(OUT/'repeated_response_summary.csv',index=False)
save_json('repeat_inventory.json',dict(total_pairs=len(pairs),counts=pairs.tier.value_counts().to_dict(),nominal_pairs=len(pairs)*50*3))
print('Repeated-item pairs:',len(pairs))
