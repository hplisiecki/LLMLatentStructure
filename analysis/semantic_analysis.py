"""SSD of signed item--PC1 associations (Section 4 and Appendix C)."""
from common import *
import argparse,sys
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(ROOT/'third_party'))
parser=argparse.ArgumentParser()
parser.add_argument('--embeddings',type=Path,required=True,help='Gensim KeyedVectors GloVe/Dolma 300d file')
args=parser.parse_args();KV_PATH=args.embeddings
def fit_ssd(target):
    from ssdiff import load_embeddings,normalize_kv,load_spacy,load_stopwords,preprocess_texts,build_docs_from_preprocessed,pca_sweep,SSD
    print('Loading and normalizing embeddings',flush=True)
    kv=normalize_kv(load_embeddings(str(KV_PATH)),l2=True,abtt_m=1)
    nlp=load_spacy('en_core_web_lg')
    assert nlp is not None
    stop=load_stopwords('en')
    save_json('ssd_environment.json',dict(spacy_model=nlp.meta,embedding_file=KV_PATH.name,embedding_vocabulary=len(kv),embedding_dimensions=kv.vector_size,stopwords=sorted(stop)))
    texts=target.item.tolist();y=target.r_pc1.to_numpy()
    pre=preprocess_texts(texts,nlp,stop);docs=build_docs_from_preprocessed(pre)
    print('Starting K=2,4,...,120 sweep',flush=True)
    sel=pca_sweep(kv=kv,docs=docs,y=y,lexicon=[],use_full_doc=True,pca_k_values=list(range(2,121,2)),window=3,sif_a=1e-3,k_min=2,k_max=5,auck_radius=3,save_figures=True,save_tables=True,out_dir=str(OUT),prefix='ssd',verbose=True)
    ssd=SSD(kv=kv,docs=docs,y=y,lexicon=[],use_full_doc=True,l2_normalize_docs=True,N_PCA=sel.best_k,use_unit_beta=True,window=3,sif_a=1e-3)
    np.save(OUT/'ssd_semantic_gradient.npy',ssd.beta_unit)
    scores=ssd.ssd_scores(include_all=True).sort_values('doc_index').reset_index(drop=True)
    assert len(scores)==len(target)
    # SSD's exported X beta omits centering offset. Export centered predictions explicitly.
    scores['prediction_centered_std']=scores.yhat_std-scores.loc[scores.kept,'yhat_std'].mean()
    scores['prediction_centered_r']=ssd.y_kept.mean()+ssd.y_kept.std()*scores.prediction_centered_std
    result=pd.concat([target,scores],axis=1)
    result.to_csv(OUT/'ssd_item_scores.csv',index=False)
    np.testing.assert_allclose(np.corrcoef(result.loc[result.kept,'r_pc1'],result.loc[result.kept,'prediction_centered_r'])[0,1]**2,ssd.r2,atol=1e-10)
    result.loc[~result.kept,['questionnaire','item']].to_csv(OUT/'ssd_embedding_exclusions.csv',index=False)
    ssd.top_words(n=40,verbose=False).to_csv(OUT/'ssd_top_words.csv',index=False)
    clusters,members=ssd.cluster_neighbors(topn=100,k=None,k_min=2,k_max=8,verbose=False)
    clusters.to_csv(OUT/'ssd_clusters.csv',index=False);members.to_csv(OUT/'ssd_cluster_members.csv',index=False)
    snips=ssd.cluster_snippets(pre_docs=pre,side='both',top_per_cluster=10)
    pd.concat([frame.assign(side=side) for side,frame in snips.items() if frame is not None and len(frame)],ignore_index=True).to_csv(OUT/'ssd_cluster_snippets.csv',index=False)
    fit=dict(input_items=len(target),retained_items=ssd.n_kept,K=int(sel.best_k),r=float(ssd.y_corr_pred),r2=float(ssd.r2),adjusted_r2=float(ssd.r2_adj))
    save_json('ssd_fit_summary.json',fit)
    print('SSD FIT',json.dumps(fit),flush=True)
    print('SSD CLUSTERS',clusters.to_string(index=False),flush=True)

with threadpool_limits(limits=4):
    fit_ssd(pd.read_csv(OUT/'item_pc1_correlations.csv'))
