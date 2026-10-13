"""Export item examples selected by numerical rank."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'
TABLE=ROOT/'results/tables'
TABLE.mkdir(parents=True,exist_ok=True)
pi=pd.read_csv(OUT/'variance_ratios.csv')
r=pd.read_csv(OUT/'item_pc1_correlations.csv')
short={
'Balanced Inventory of Desirable Responding Version 6':'BIDR',
'DERS (Difficulties in Emotion Regulation Scale)':'DERS',
'Basic Psychological Needs Scale':'Basic Needs',
'Five Facet Mindfullness Questionnaire (FFMQ)':'FFMQ',
'CERQ (Cognitive Emotion Regulation Questionnaire)':'CERQ',
'Individualism-collectivism IND-COL':'IND-COL',
'BIS_BAS Scales':'BIS/BAS',
'Adult Temperament Questionnaire':'ATQ',
'Psychological Well-Being Scales':'PWB',
'Schwartz_Values':'Schwartz Values',
'Meaning in Life Questionnaire (MLQ)':'MLQ',
'Interpersonal Reactivity Index (IRI)':'IRI',
'The Internal Representation Questionnaire':'Internal Representation',
'ECR-R (Experiences in Close Relationships–Revised)':'ECR-R',
'Varieties of Inner Speech Questionnaire-R':'VISQ-R',
'Self-Monitoring Scale-Revised':'Self-Monitoring',
}
def esc(v):
 s=str(v).strip().replace('‚',',')
 return s.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
def qname(q):return esc(short.get(q.strip(),q.strip()))
def table_start(caption,label):
 return (r'\begin{table*}[!t]'+'\n'+r'\centering\small'+'\n'+
 r'\begin{tabular}{@{}p{.19\textwidth}p{.69\textwidth}r@{}}'+'\n'+r'\toprule'+'\n')
parts=[]
parts.append(table_start('',''))
parts.append(r'Questionnaire & Item & $r_i$ \\'+'\n'+r'\midrule'+'\n')
selected=[]
for heading,sub in [('Strongest positive associations',r.nlargest(10,'r_pc1')),('Strongest negative associations',r.nsmallest(10,'r_pc1'))]:
 parts.append(r'\multicolumn{3}{l}{\textit{'+heading+r'}} \\'+'\n')
 selected.append(sub)
 for _,row in sub.iterrows():
  parts.append(f'{qname(row.questionnaire)} & {esc(row["item"])} & ${row.r_pc1:+.3f}$ '+r'\\'+'\n')
 parts.append(r'\addlinespace'+'\n')
parts.append(r'''\bottomrule
\end{tabular}
\caption{The ten strongest positive and ten strongest negative item--PC1
correlations under neutral prompting, ranked separately by signed correlation.
Eligible items have at least 15 valid model responses and nonzero response
variance. Wording is shown as administered; questionnaire abbreviations follow
Appendix~\ref{app:collection}. Signs use the reference PC1 orientation.}
\label{tab:pc1_item_examples}
\end{table*}
''')
(TABLE/'pc1_item_examples.tex').write_text(''.join(parts),encoding='utf-8')
pd.concat(selected).to_csv(OUT/'pc1_item_examples.csv',index=False)
top=pi.nlargest(20,'variance_ratio')
parts=[table_start('',''),r'Questionnaire & Item & $\pi_i$ \\'+'\n'+r'\midrule'+'\n']
for _,row in top.iterrows():
 parts.append(f'{qname(row.questionnaire)} & {esc(row["item"])} & {row.variance_ratio:.2f} '+r'\\'+'\n')
parts.append(r'''\bottomrule
\end{tabular}
\caption{The 20 highest neutral-to-human-simulation variance ratios among
1,312 eligible items. Selection uses the untransformed ratio, with at least
five valid responses in each condition and positive human-simulation variance.
The ratio measures relative disagreement, not the direction of an item's
association with PC1.}
\label{tab:ratio_item_examples}
\end{table*}
''')
(TABLE/'ratio_item_examples.tex').write_text(''.join(parts),encoding='utf-8')
top.to_csv(OUT/'ratio_item_examples.csv',index=False)
