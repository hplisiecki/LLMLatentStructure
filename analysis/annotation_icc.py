"""Fixed-panel consistency ICC(3,1) and ICC(3,3), with F-based 95% CIs.
Each rubric uses its own complete three-rater items. Scores are treated as interval.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import f
ROOT=Path(__file__).resolve().parents[1]
d=pd.read_csv(ROOT/'data/annotations/ratings.csv')
rubrics=['inner_life_content','observable_behavior','beliefs_judgments','human_world_dependence','valence','social_desirability']
rows=[]
for rubric in rubrics:
    wide=d.pivot(index='item_id',columns='annotator_id',values=rubric)
    a=wide.dropna().to_numpy();n,k=a.shape
    assert k==3
    grand=a.mean(); item=a.mean(axis=1,keepdims=True);rater=a.mean(axis=0,keepdims=True)
    ms_item=k*np.square(item-grand).sum()/(n-1)
    error=a-item-rater+grand
    ms_error=np.square(error).sum()/((n-1)*(k-1))
    F=ms_item/ms_error;df1=n-1;df2=(n-1)*(k-1)
    lower_F=F/f.ppf(.975,df1,df2);upper_F=F*f.ppf(.975,df2,df1)
    def single(x):return (x-1)/(x+k-1)
    def average(x):return 1-1/x
    # Independent consistency identity: alpha across the three centered raters.
    alpha=k/(k-1)*(1-a.var(axis=0,ddof=1).sum()/a.sum(axis=1).var(ddof=1))
    assert np.isclose(alpha,average(F))
    rows.append(dict(rubric=rubric,n_items=n,excluded_items=len(wide)-n,n_raters=k,
        icc_3_1=single(F),icc_3_1_low=single(lower_F),icc_3_1_high=single(upper_F),
        icc_3_3=average(F),icc_3_3_low=average(lower_F),icc_3_3_high=average(upper_F),
        ms_item=ms_item,ms_error=ms_error,F=F,df1=df1,df2=df2))
out=pd.DataFrame(rows)
out.to_csv(ROOT/'results/annotation_icc.csv',index=False)
print(out[['rubric','n_items','icc_3_1','icc_3_3','icc_3_3_low','icc_3_3_high']].round(4).to_string(index=False))
