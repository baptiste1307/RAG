import bm25s
from .models import TextChunk
from .config import global_state as state, load_json


def _validate_retrieval_args(question: str, k: int) -> None:
    """Validate retrieval inputs before loading the BM25 index.

    This keeps bad CLI arguments from turning into confusing low-level errors
    inside `bm25s`. The user should see a simple message explaining what is
    wrong with the question or with `k`.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Invalid query: expected a non-empty string.")
    if not isinstance(k, int):
        raise ValueError("Invalid --k: expected an integer such as --k 10.")
    if k <= 0:
        raise ValueError("Invalid --k: value must be greater than 0.")


def _get_best_chunks_index(
    question: str, k: int
) -> tuple[list[list[int]], list[list[float]]]:
    """Return the best chunks indexes to answer the query"""
    try:
        # load index from index_dir
        retriever = bm25s.BM25.load(state.save_index_path)
        # tokenize the query
        query_tokens = bm25s.tokenize(question)
        # retrieve best chunks to answer the query
        # result = index of best chunks in corpus
        # scores = relevance score for each result
        results, scores = retriever.retrieve(query_tokens, k=k)
        # ex: [[0, 21, 65]], [[12.7, 9.4, 6.1]]
        # => chunk 0 has a score of 12.7 (this score is not % !!)
        return results, scores
    except Exception as e:
        raise Exception(
            "Error while getting best chunks indexes for "
            f"question: '{question}': {e}"
        )


def get_best_chunks(question: str, k: int) -> list[TextChunk]:
    """With the best chunk indexes of corpus, return a list of
    the corresponding TextChunk elements from chunks"""

    try:
        _validate_retrieval_args(question, k)

        # if dataset not already indexed
        if not state.save_index_path.exists():
            raise ValueError(
                "Save_index_path does not exists: index dataset first."
            )

        # load all chunks
        chunks = [
            TextChunk.model_validate(chunk)
            for chunk in load_json(state.save_chunks_path)
        ]
        # results = best chunks indexes
        results, _ = _get_best_chunks_index(question, k)

        # with those indexes, we find the corresponding chunks in 'chunks'
        return [chunks[index] for index in results[0][:k]]
    except Exception as e:
        raise Exception(
            "Error while getting best chunks for "
            f"question: '{question}': {e}"
        )
