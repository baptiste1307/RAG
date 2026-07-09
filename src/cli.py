import fire
from .indexing import index_repo
from .retrieval import get_best_chunks
from .build_results import (
    build_student_search_results,
    build_student_search_results_and_answer,
)
from .generation import build_prompt, get_llm_answer
from pathlib import Path
from .evaluation import evaluate_search_results
from .models import TextChunk


def _validate_k(k: int) -> int:
    """Check that k is a useful positive number before doing any work.

    `k` means "how many chunks should be retrieved". A negative number, zero,
    or a non-integer value does not make sense, so this helper raises a clear
    error before the retriever is called.
    """
    if not isinstance(k, int):
        raise ValueError("Invalid --k: expected an integer such as --k 10.")
    if k <= 0:
        raise ValueError("Invalid --k: value must be greater than 0.")
    return k


class CLI:
    """Expose the RAG pipeline as simple terminal commands.

    Python Fire reads this class and turns each public method into a command.
    For example, the `index` method becomes `python -m src index`. This keeps
    the command-line interface small: the CLI only converts strings to paths
    and delegates the real work to the indexing, retrieval, generation, and
    evaluation modules.
    """

    def index(self, max_chunk_size: int = 2000) -> str:
        """Build the searchable knowledge base from the vLLM repository.

        This command reads the repository, splits Python and Markdown files into
        chunks, builds the BM25 index, and saves the generated files under
        `data/processed`. Run it before searching when no index exists yet.
        """
        if max_chunk_size <= 0 or max_chunk_size > 2000:
            raise ValueError(
                "Invalid --max_chunk_size: expected a value from 1 to 2000."
            )
        return index_repo(max_chunk_size)

    # search for a single query
    def search(self, query: str, k: int = 10) -> list[TextChunk]:
        """Return the raw chunks retrieved for one question.

        This command does not write a natural-language answer. It is like
        asking the search engine: "Which parts of the repository look useful
        for this question?" The result is a list of `TextChunk` objects.
        """
        return get_best_chunks(query, _validate_k(k))

    # Process multiple questions and output search results
    def search_dataset(
        self, dataset_path: str, save_directory: str, k: int = 10
    ) -> str:
        """Retrieve sources for every question in a dataset and save JSON.

        The input file contains many questions. For each one, this command
        stores the top-k retrieved sources in the format expected by the
        evaluator. The output file is saved inside `save_directory`.
        """
        return build_student_search_results(
            Path(dataset_path), Path(save_directory), _validate_k(k)
        )

    # Answer a single question using indexed documents
    # can be executed directly, without search before
    def answer(self, question: str, k: int = 10) -> str:
        """Generate one readable answer for a single question.

        This command first retrieves relevant chunks, then builds a context
        prompt, then returns one concise final answer. Use this when you want a
        user-facing answer instead of the raw search chunks.
        """
        best_chunks = get_best_chunks(question, _validate_k(k))
        prompt = build_prompt(best_chunks, question)
        return get_llm_answer(prompt)

    # Generate answers from search results
    # Uses previously generated StudentSearchResults
    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> str:
        """Generate answers from an existing search-results file.

        This command expects a JSON file produced by `search_dataset`. It uses
        the already retrieved sources for each question, generates an answer,
        and saves a `StudentSearchResultsAndAnswer` JSON file.
        """
        return build_student_search_results_and_answer(
            Path(student_search_results_path), Path(save_directory)
        )

    # Evaluate search results against ground truth
    def evaluate_student_search_results(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000,
    ) -> dict[str, float]:
        """Evaluate retrieved sources with the moulinette-style command name.

        The evaluator compares each retrieved source with the expected source
        ranges from an answered dataset. It returns recall@1, recall@3,
        recall@5, and recall@k.
        """
        return evaluate_search_results(
            Path(student_answer_path),
            Path(dataset_path),
            _validate_k(k),
            max_context_length,
        )

    # same but dont know the exact command name the moulinette will use
    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 10,
        max_context_length: int = 2000,
    ) -> dict[str, float]:
        """Evaluate retrieved sources with a shorter local command name.

        This is the same as `evaluate_student_search_results`, but easier to
        type when testing locally: `python -m src evaluate ...`.
        """
        return evaluate_search_results(
            Path(student_answer_path),
            Path(dataset_path),
            _validate_k(k),
            max_context_length,
        )


if __name__ == "__main__":
    fire.Fire(CLI)
