from __future__ import annotations

from mteb.abstasks.AbsTaskBitextMining import AbsTaskBitextMining
from mteb.abstasks.MultilingualTask import MultilingualTask
from mteb.abstasks.TaskMetadata import TaskMetadata

_LANGUAGES = {
    "de-en": ["deu-Latn", "eng-Latn"],
    "fr-en": ["fra-Latn", "eng-Latn"],
    "ru-en": ["rus-Cyrl", "eng-Latn"],
    "zh-en": ["cmn-Hans", "eng-Latn"],
}


_SPLITS = ["test"]


class BUCCBitextMiningFast(AbsTaskBitextMining, MultilingualTask):
    fast_loading = True
    metadata = TaskMetadata(
        name="BUCC.v2",
        dataset={
            "path": "mteb/bucc-bitext-mining",
            "revision": "1739dc11ffe9b7bfccd7f3d585aeb4c544fc6677",
        },
        description="BUCC bitext mining dataset",
        reference="https://comparable.limsi.fr/bucc2018/bucc2018-task.html",
        type="BitextMining",
        category="s2s",
        modalities=["text"],
        eval_splits=_SPLITS,
        eval_langs=_LANGUAGES,
        main_score="f1",
        date=("2017-01-01", "2018-12-31"),
        domains=["Written"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="human-annotated",
        dialect=[],
        sample_creation="human-translated",
        bibtex_citation=r"""
@inproceedings{zweigenbaum-etal-2017-overview,
  abstract = {This paper presents the BUCC 2017 shared task on parallel sentence extraction from comparable corpora. It recalls the design of the datasets, presents their final construction and statistics and the methods used to evaluate system results. 13 runs were submitted to the shared task by 4 teams, covering three of the four proposed language pairs: French-English (7 runs), German-English (3 runs), and Chinese-English (3 runs). The best F-scores as measured against the gold standard were 0.84 (German-English), 0.80 (French-English), and 0.43 (Chinese-English). Because of the design of the dataset, in which not all gold parallel sentence pairs are known, these are only minimum values. We examined manually a small sample of the false negative sentence pairs for the most precise French-English runs and estimated the number of parallel sentence pairs not yet in the provided gold standard. Adding them to the gold standard leads to revised estimates for the French-English F-scores of at most +1.5pt. This suggests that the BUCC 2017 datasets provide a reasonable approximate evaluation of the parallel sentence spotting task.},
  address = {Vancouver, Canada},
  author = {Zweigenbaum, Pierre  and
Sharoff, Serge  and
Rapp, Reinhard},
  booktitle = {Proceedings of the 10th Workshop on Building and Using Comparable Corpora},
  doi = {10.18653/v1/W17-2512},
  editor = {Sharoff, Serge  and
Zweigenbaum, Pierre  and
Rapp, Reinhard},
  month = aug,
  pages = {60--67},
  publisher = {Association for Computational Linguistics},
  title = {Overview of the Second {BUCC} Shared Task: Spotting Parallel Sentences in Comparable Corpora},
  url = {https://aclanthology.org/W17-2512},
  year = {2017},
}
""",
        adapted_from=["BUCC"],
    )
