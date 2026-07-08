from .models import MinimalSource, StudentSearchResults, RagDataset
from .config import load_json
from pathlib import Path


def source_overlap_ratio(
    expected: MinimalSource,
    retrieved: MinimalSource,
) -> float:
    """Measure how much a retrieved source covers an expected source.

    The evaluator does not require the two source ranges to be exactly equal.
    It only asks whether they overlap enough in the same file. This function
    returns the fraction of the expected range that is covered by the retrieved
    range. If the files are different, the answer is always 0.
    """

    if expected.file_path != retrieved.file_path:
        return 0.0

    # compare text intervals to find common part lenght
    start = max(
        expected.first_character_index, retrieved.first_character_index
    )
    end = min(expected.last_character_index, retrieved.last_character_index)
    # if retrieved part outside expected part: negative overlap -> take 0
    # if retrieved part inside expected part: positive overlap.
    # Ex: expected indexed: 100-200. Retrieved indexes: 180-220. => overlap: 20
    overlap = max(0, end - start)

    expected_len = max(
        0, expected.last_character_index - expected.first_character_index
    )

    if expected_len <= 0:
        return 0.0

    # 20/100 = 0.2
    return overlap / expected_len


def source_was_found(
    expected: MinimalSource,
    retrieved_sources: list[MinimalSource],
    min_overlap: float = 0.05,
) -> bool:
    """Return True when one retrieved source matches an expected source.

    A source is considered found if it is in the same file as the expected
    source and if the character ranges overlap by at least `min_overlap`.
    The subject uses 0.05, meaning 5 percent overlap is enough.
    """
    for retrieved in retrieved_sources:
        if source_overlap_ratio(expected, retrieved) >= min_overlap:
            return True
    return False


def recall_at_k(
    expected_sources: list[MinimalSource],
    retrieved_sources: list[MinimalSource],
    k: int,
) -> float:
    """Compute recall@k for one question.

    Recall@k answers a simple question: "Among the correct sources, how many
    did we find in the first k retrieved sources?" A score of 1.0 means every
    expected source was found. A score of 0.0 means none were found.
    """

    if not expected_sources:
        return 0.0

    found = 0
    # count how many expected sources are find in the first k retrieved_sources
    for expected in expected_sources:
        if source_was_found(expected, retrieved_sources[:k]):
            found += 1
            continue

    # Ex: 3 sources found out of 10 => 0.3
    return found / len(expected_sources)


def average(values: list[float]) -> float:
    """Return the mean of a list, or 0.0 when the list is empty.

    This avoids a division-by-zero crash when no question could be evaluated.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def evaluate_search_results(
    student_answer_path: Path,
    # corrected dataset, contains:
    # question_id, question, expected sources, answer
    dataset_path: Path,
    k: int,
    max_context_length: int,
) -> dict[str, float]:
    """Evaluate one student search-results file against ground truth.

    The student file contains the sources retrieved by the RAG system. The
    answered dataset contains the correct source ranges. This function matches
    questions by `question_id`, computes recall@1, recall@3, recall@5, and
    recall@k for each question, then returns the average scores.
    """

    if not student_answer_path.exists():
        raise Exception(
            "StudentSearchResults does not exists: search dataset first."
        )

    answered_questions = RagDataset.model_validate(load_json(dataset_path))

    ssr = StudentSearchResults.model_validate(load_json(student_answer_path))

    scores_1 = []
    scores_3 = []
    scores_5 = []
    scores_k = []

    results_by_id = {
        result.question_id: result for result in ssr.search_results
    }

    for answered_question in answered_questions.rag_questions:
        if not hasattr(answered_question, "sources"):
            continue
        student_result = results_by_id.get(answered_question.question_id)
        if student_result is None:
            continue

        scores_1.append(
            recall_at_k(
                answered_question.sources, student_result.retrieved_sources, 1
            )
        )
        scores_3.append(
            recall_at_k(
                answered_question.sources, student_result.retrieved_sources, 3
            )
        )
        scores_5.append(
            recall_at_k(
                answered_question.sources, student_result.retrieved_sources, 5
            )
        )
        scores_k.append(
            recall_at_k(
                answered_question.sources, student_result.retrieved_sources, k
            )
        )

    return {
        "recall@1": average(scores_1),
        "recall@3": average(scores_3),
        "recall@5": average(scores_5),
        f"recall@{k}": average(scores_k),
    }
