Data dictionary
===============

``data/responses.csv``
    One questionnaire/item-index/model/condition administration per row.
    ``response`` is the parsed numeric response; missing values denote responses
    without a usable numeric score. ``response_raw`` preserves the returned
    text. Conditions are ``neutral``, ``human_simulation``, and ``llm_analog``.
    ``item_index`` is local to a questionnaire, not a global item identifier.

``data/questionnaires.json`` and ``data/questionnaire_registry.csv``
    Questionnaire item text, instructions, response options, and selection/scale
    metadata. ``analysis/prompts.py`` defines the three prompt constructors.
    The archive contains 47 administered instruments; ``included_questionnaires.csv``
    is the authoritative list of 45 retained for the shared-structure analyses.
    IRQ and unrevised VISQ are excluded for overlap. MFQ parts are combined
    under their instrument label in the response archive.

``data/factor_loadings/``
    One item-text-indexed matrix per included questionnaire and condition,
    plus ``summary.csv`` with extraction method, sample size, and factor count.
    The first numeric column supplies the first-factor loading or first-PC
    coefficient. Absolute first loadings are used only for magnitude changes;
    signed first loadings are used for questionnaire scores.

``data/model_splits.csv``
    Each row is a permutation of positions 0 through 49 in the alphabetically
    sorted model list. The first 25 and last 25 define disjoint split halves.

``data/annotations/ratings.csv``
    Long-form ratings by item and anonymous rater, with presentation metadata.
    The six rubric columns are ``inner_life_content``, ``observable_behavior``,
    ``beliefs_judgments``, ``human_world_dependence``, ``valence``, and
    ``social_desirability``. Missing cells remain missing. The handbook defines
    the rating scales. Three psychology-student interns rated the items without
    knowing the hypothesis after a meeting using the 20-item training set;
    they knew the ratings were for the study and were not compensated.

``data/annotations/eligible_items.csv``
    Item identifiers and questionnaire/text keys for the annotation sampling
    frame. IDs link the ratings and manifests to item text; these IDs are not
    response-archive item indices.

``data/annotations/production_manifest.csv``
    Annotation assignment and presentation inventory, including assignments
    outside the completed rating sample. It is not a table of observed ratings.

``data/annotations/training_sample_manifest.csv``
    The 20 items used in the introductory annotation exercise.

Output conventions
------------------
Most item outputs join on questionnaire and exact item text; ratio outputs
also retain item index. Model outputs use the full provider/model identifier.
CSV files contain full-precision statistics unless the analysis map specifies
rounding. JSON summaries describe analysis populations and computed statistics.
