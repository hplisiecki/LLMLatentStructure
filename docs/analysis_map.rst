Analysis map
============

Questionnaire structure (Section 3; Appendix B)
-----------------------------------------------
``global_structure.py`` averages repeated item text within questionnaire/model,
standardizes items across models, weights by the first loading, and standardizes
questionnaire summaries before global PCA. Missing standardized responses are
filled with zero; models require at least floor(0.8 * number of questionnaires)
observed summaries, and missing retained summaries are filled with zero.
The neutral matrix contains 50 models and 45 questionnaires.
The sign is oriented so the item "I often act on the spur of the moment." has
a negative association. Outputs include ``global_pca_*``,
``questionnaire_model_scores.csv``, and ``condition_comparison.json``.
``extraction_variance.py`` produces ``neutral_extraction_variance.csv``.
``item_correlations.py`` correlates responses with neutral PC1, requiring at
least 15 valid model responses and nonconstant items. It produces 1,325 item
associations and records exclusions separately.

Semantic analysis (Section 4; Appendix C)
-----------------------------------------
``semantic_analysis.py`` fits SSD to signed item--PC1 correlations from the
same 45-questionnaire population. Text preprocessing leaves 1,324 usable items.
Embeddings are L2-normalized with one all-but-the-top component removed; document
representations use SIF weighting and L2 normalization. The PCA sweep selects
16 dimensions, followed by regression and semantic-neighbor clustering.
``ssd_fit_summary.json``, ``ssd_item_scores.csv``, ``ssd_clusters.csv``, and
associated member/snippet/word tables describe this single fitted analysis.
The fit R-squared is in-sample, not held-out prediction performance.

Respondent framing (Section 5; Appendix D)
------------------------------------------
``framing_analysis.py`` forms neutral/human-simulation variance ratios for
1,312 items with at least five responses per condition and nonzero denominator
variance. Exported ratios and variances are rounded to three decimals before
subsequent primary scoring. Item-index observations are retained for ratios;
loading matrices and model scores average repeated text within questionnaire.
The main loading-change comparison uses log(1 + ratio) after a 99th-percentile
cap and the change in absolute first loading (human minus neutral).

Variance compression uses 1,310 items with positive variance and at least five
responses in all three conditions. Split-half reliability uses the 500 fixed
partitions in ``data/model_splits.csv``. Alternative scores use unrounded log
ratios centered within questionnaire, and negative residuals of log human
variance regressed on log neutral variance. Their common loading sample has
1,291 observations. Outputs are ``variance_*``, ``loading_change_*``,
``split_half_*``, and ``alternative_ratio_*``.

The secondary model score uses positive log-ratio weights capped at the 99th
percentile, population-standardized neutral responses, and the sum of available
weights as denominator. Duplicate text weights retain the last entry in the
ratio-sorted table. The analog comparison applies these same weights to
responses standardized within the analog condition. Its .482 rank correlation
is a comparison of weighted scores; it is not a comparison of global PC1s.

``repeated_responses.py`` inventories duplicate texts in the administered
response archive, including the two instruments excluded from global PCA.
It exports ``repeat_inventory.json``, ``repeated_item_pairs.csv``, and
``repeated_response_summary.csv``. The 139 observed exact-prompt pairs are
model-by-repeated-item combinations with both responses present, out of 150.

Human assessment (Section 6; Appendix E)
----------------------------------------
``annotation_reliability.py`` computes rubric means, ordinal Krippendorff alpha
with item bootstrap intervals (2,000 replicates), pairwise agreement, and
rating distributions. ``annotation_icc.py`` computes fixed-panel consistency
ICC(3,1) and ICC(3,3), using complete ratings within each rubric.
``annotation_associations.py`` joins item ratings to full-precision PC1
associations: 289 matched items across 44 questionnaires, with 281 complete
three-rater items. Correlation intervals resample questionnaires (5,000
replicates); regressions use Fisher-transformed outcomes, standardized
predictors, and questionnaire-clustered uncertainty. Outputs are ``pc1_*``,
``reliability.csv``, ``annotation_icc.csv``, and the rater/agreement tables.
These analyses hold the fitted component and annotator panel fixed.

Figures and tables
------------------
``make_figures.py`` writes figures and model bootstrap results; bootstrap
resampling uses 1,000 replicates and seed 42. ``make_tables.py`` writes item
examples and LaTeX fragments. The repository does not contain a paper build.
