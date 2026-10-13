# LLM psychometric response structure

Data and analysis code for the shared latent structure of LLM psychometric
responses. This repository covers questionnaire summaries and global PCA,
semantic analysis, respondent framing, and human semantic ratings.
The manuscript is maintained separately.

## Quick start

Use Python 3.12. From this directory:

```sh
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python run_analysis.py
python validation/check_results.py
```

On macOS/Linux, activate with `source .venv/bin/activate`.
For the exact tested dependency set, install `requirements-lock.txt` instead
of `requirements.txt`. The spaCy language model remains an optional separate
installation for SSD.

The standard run needs no model API access. It regenerates numerical results,
figures, and LaTeX table fragments under `results/`. Running it replaces those
outputs. Included SSD outputs remain available without the external embeddings.

To also recompute semantic analysis, install the spaCy English model
`en_core_web_lg` version 3.8.0, then provide the Gensim KeyedVectors file:

```sh
python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
python run_analysis.py --embeddings /path/to/dolma_300_2024_1.2M.100_combined.kv
```

Use the GloVe/Dolma 300-dimensional, 1.2-million-word embedding resource.
Keep its `.kv.vectors.npy` sidecar beside the `.kv` file. These large assets
are external to this repository; substituting embeddings changes the analysis.
See `docs/embedding_assets.json` for the exact asset fingerprints.
SSD evaluates PCA dimensions 2, 4, ..., 120 and chooses the dimension using its
joint interpretability/stability criterion. The reported fit uses 16 dimensions.

## Repository contents

* `analysis/`: executable analysis scripts and exact prompt constructors.
* `data/`: response archive, questionnaire definitions, retained-instrument
  list, fixed loading matrices, model partitions, and anonymized ratings.
* `annotations/handbook.pdf`: annotation instructions.
* `results/`: machine-readable results; `figures/` and `tables/` contain
  publication assets.
* `docs/analysis_map.rst`: analyses, assumptions, and corresponding outputs.
* `docs/data_dictionary.rst`: file schemas and joining conventions.
* `validation/`: numerical checks and input fingerprints.
* `third_party/ssdiff/`: required SSD runtime and its license/source notice.

## Reproducibility

The default pipeline uses the supplied questionnaire loading matrices. Optional
`python analysis/factor_extraction.py` refits them into
`results/refitted_loadings/`. Parallel analysis is unseeded, so a refit can
select different factor counts; it does not replace the supplied matrices.
First-column loadings define questionnaire summaries and loading-change checks.
The SSD regression is exploratory and in-sample; annotations and framing provide
separate, complementary interpretive evidence.

`validation/check_results.py` checks reported quantities after a run. The
verification report distinguishes numerical checks from rerunning SSD.
Input integrity can be checked with `python validation/check_inputs.py`.
Third-party SSD code remains subject to the included MIT license.
