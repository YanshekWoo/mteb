from __future__ import annotations

from mteb.abstasks.TaskMetadata import TaskMetadata

from ....abstasks.AbsTaskSTS import AbsTaskSTS


class BiossesSTS(AbsTaskSTS):
    metadata = TaskMetadata(
        name="BIOSSES",
        dataset={
            "path": "mteb/biosses-sts",
            "revision": "d3fb88f8f02e40887cd149695127462bbcf29b4a",
        },
        description="Biomedical Semantic Similarity Estimation.",
        reference="https://tabilab.cmpe.boun.edu.tr/BIOSSES/DataSet.html",
        type="STS",
        category="s2s",
        modalities=["text"],
        eval_splits=["test"],
        eval_langs=["eng-Latn"],
        main_score="cosine_spearman",
        date=None,
        domains=["Medical"],
        task_subtypes=[],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=r"""
@article{10.1093/bioinformatics/btx238,
  abstract = {{The amount of information available in textual format is rapidly increasing in the biomedical domain. Therefore, natural language processing (NLP) applications are becoming increasingly important to facilitate the retrieval and analysis of these data. Computing the semantic similarity between sentences is an important component in many NLP tasks including text retrieval and summarization. A number of approaches have been proposed for semantic sentence similarity estimation for generic English. However, our experiments showed that such approaches do not effectively cover biomedical knowledge and produce poor results for biomedical text.We propose several approaches for sentence-level semantic similarity computation in the biomedical domain, including string similarity measures and measures based on the distributed vector representations of sentences learned in an unsupervised manner from a large biomedical corpus. In addition, ontology-based approaches are presented that utilize general and domain-specific ontologies. Finally, a supervised regression based model is developed that effectively combines the different similarity computation metrics. A benchmark data set consisting of 100 sentence pairs from the biomedical literature is manually annotated by five human experts and used for evaluating the proposed methods.The experiments showed that the supervised semantic sentence similarity computation approach obtained the best performance (0.836 correlation with gold standard human annotations) and improved over the state-of-the-art domain-independent systems up to 42.6\\% in terms of the Pearson correlation metric.A web-based system for biomedical semantic sentence similarity computation, the source code, and the annotated benchmark data set are available at: http://tabilab.cmpe.boun.edu.tr/BIOSSES/.}},
  author = {Soğancıoğlu, Gizem and Öztürk, Hakime and Özgür, Arzucan},
  doi = {10.1093/bioinformatics/btx238},
  eprint = {https://academic.oup.com/bioinformatics/article-pdf/33/14/i49/50315066/bioinformatics\_33\_14\_i49.pdf},
  issn = {1367-4803},
  journal = {Bioinformatics},
  month = {07},
  number = {14},
  pages = {i49-i58},
  title = {{BIOSSES: a semantic sentence similarity estimation system for the biomedical domain}},
  url = {https://doi.org/10.1093/bioinformatics/btx238},
  volume = {33},
  year = {2017},
}
""",
    )

    @property
    def metadata_dict(self) -> dict[str, str]:
        metadata_dict = super().metadata_dict
        metadata_dict["min_score"] = 0
        metadata_dict["max_score"] = 5
        return metadata_dict
