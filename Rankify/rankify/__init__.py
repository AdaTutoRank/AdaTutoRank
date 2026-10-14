"""
Rankify (reranker-only release): the rerankers compared in AdaTutoRank, the
basic-RAG generator and QA metrics used to evaluate them.

    from rankify.models.reranking import Reranking
    reranker = Reranking.from_config("adatutorank-8b")   # configs/rerankers.yaml
    reranker.rank(documents)
"""

__version__ = "0.1.5"
