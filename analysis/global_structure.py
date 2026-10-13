"""Questionnaire summaries and the shared model component (Section 3)."""
from common import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def compute(condition):
    r=responses();r=r[(r.condition==condition)&r.response.notna()]
    scores={}
    for p in sorted((DATA/'factor_loadings').glob('*__'+condition+'.csv')):
        q=p.name.removesuffix('__'+condition+'.csv')
        w=pd.read_csv(p,index_col=0).iloc[:,0]
        m=r[r.questionnaire==q].pivot_table(index='model',columns='item',values='response')
        common=w.index.intersection(m.columns,sort=False)
        if len(common)<3:continue
        z=(m[common]-m[common].mean())/(m[common].std()+1e-9)
        scores[q]=pd.Series(z.fillna(0).to_numpy()@w.loc[common].to_numpy(),index=z.index)
    s=pd.DataFrame(scores);s=s.dropna(thresh=int(.8*len(s.columns))).fillna(0)
    x=StandardScaler().fit_transform(s);pca=PCA(svd_solver='full');pc=pca.fit_transform(x)
    # Anchor the arbitrary sign using a specified item, before semantic analyses.
    anchor=r[r['item'].eq('I often act on the spur of the moment.')].groupby('model').response.mean()
    if pd.Series(pc[:,0],index=s.index).corr(anchor)>0:pc[:,0]*=-1
    return s,x,pc,pca.explained_variance_ratio_*100
s,x,pc,e=compute('neutral')
assert s.shape==(50,45)
pd.DataFrame({'model':s.index,'PC1':pc[:,0],'PC2':pc[:,1]}).sort_values('PC1').to_csv(OUT/'global_pca_model_scores.csv',index=False)
s.to_csv(OUT/'questionnaire_model_scores.csv')
c=pd.DataFrame({'questionnaire':s.columns,'r_pc1':[np.corrcoef(x[:,i],pc[:,0])[0,1] for i in range(x.shape[1])]})
c['abs_r_pc1']=c.r_pc1.abs();c.sort_values('abs_r_pc1',ascending=False).to_csv(OUT/'global_pca_questionnaire_correlations.csv',index=False)
pd.DataFrame({'component':np.arange(1,len(e)+1),'variance_percent':e,'cumulative_percent':np.cumsum(e)}).to_csv(OUT/'global_pca_spectrum.csv',index=False)
print('Global PC1:',e[0],'percent; models/questionnaires:',s.shape)
if __name__=='__main__':
    a,ax,ap,ae=compute('llm_analog')
    save_json('condition_comparison.json',{'n_models':len(a),'llm_analog_pc1_variance_percent':float(ae[0])})
