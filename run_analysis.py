"""Reproduce the reported analyses from this repository's data."""
from pathlib import Path
import argparse,subprocess,sys,os
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--embeddings',type=Path,help='Also rerun SSD using the GloVe/Dolma 300d KeyedVectors file')
args=parser.parse_args()
steps=['global_structure.py','item_correlations.py','extraction_variance.py','annotation_reliability.py','annotation_icc.py','annotation_associations.py','framing_analysis.py','repeated_responses.py','make_tables.py','make_figures.py']
env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['PYTHONDONTWRITEBYTECODE']='1'
for name in steps:
    print('Running',name,flush=True)
    subprocess.run([sys.executable,str(ROOT/'analysis'/name)],cwd=ROOT,env=env,check=True)
if args.embeddings:
    subprocess.run([sys.executable,str(ROOT/'analysis/semantic_analysis.py'),'--embeddings',str(args.embeddings.resolve())],cwd=ROOT,env=env,check=True)
print('Analysis complete. Results are in results/; figures and tables are in results/figures and results/tables.')
