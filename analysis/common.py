"""Shared paths, analysis population, and response aggregation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
CONDITIONS = ['neutral', 'human_simulation', 'llm_analog']
RUBRICS = ['inner_life_content', 'observable_behavior', 'beliefs_judgments', 'human_world_dependence', 'valence', 'social_desirability']
def included():
    return set(pd.read_csv(DATA / 'included_questionnaires.csv').questionnaire)
def responses(selected=True):
    d = pd.read_csv(DATA / 'responses.csv')
    d['response'] = pd.to_numeric(d.response, errors='coerce')
    return d[d.questionnaire.isin(included())].copy() if selected else d
def loadings(condition):
    rows = []
    for p in sorted((DATA / 'factor_loadings').glob('*__' + condition + '.csv')):
        q = p.name.removesuffix('__' + condition + '.csv')
        weights = pd.read_csv(p,index_col=0)
        values = weights.iloc[:,0].abs()
        rows.extend(dict(questionnaire=q,item=item,loading=float(v)) for item,v in values.items())
    return pd.DataFrame(rows)
def save_json(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
