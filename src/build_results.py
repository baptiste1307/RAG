import json
from pathlib import Path
from tqdm import tqdm
from .config import load_json, global_state as state
from .generation import build_prompt, get_llm_answer
from .models import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    TextChunk,
)
from .retrieval import get_best_chunks


def build_dataset_best_chunks(
    questions_dataset: RagDataset, k: int
) -> list[list[TextChunk]]:
    """Retrieve the best chunks for every question in a dataset.

    Think of this as running `search` many times: one time for each question.
    The returned list keeps the same order as the dataset, so result number 0
    belongs to question number 0, result number 1 belongs to question number 1,
    and so on.
    """

    try:
        return [
            get_best_chunks(question.question, k)
            for question in questions_dataset.rag_questions
        ]
    except Exception as e:
        raise Exception(e)


def build_student_search_results(
    dataset_path: Path, save_directory: Path, k: int
) -> str:
    """Create the JSON file used to evaluate retrieval quality.

    The input dataset contains questions. For each question, this function asks
    the retriever for the top-k chunks and keeps only their source locations.
    The final JSON follows the `StudentSearchResults` model expected by the
    project and by the moulinette.
    """
    try:
        if k <= 0:
            raise ValueError("Invalid --k: value must be greater than 0.")
        # create parent directory(ies)
        save_directory.mkdir(parents=True, exist_ok=True)
        output_path = save_directory / dataset_path.name

        questions_dataset = RagDataset.model_validate(load_json(dataset_path))
        dataset_best_chunks = build_dataset_best_chunks(questions_dataset, k)

        dataset_minimal_search_results = [
            MinimalSearchResults(
                # randomly generated id
                question_id=question.question_id,
                question=question.question,
                question_str=question.question,
                retrieved_sources=[
                    chunk.source for chunk in dataset_best_chunks[index]
                ],
            )
            for index, question in enumerate(questions_dataset.rag_questions)
        ]

        ssr = StudentSearchResults(
            search_results=dataset_minimal_search_results, k=k
        )

        data = ssr.model_dump()

        # If this json file doesn't exist, it is created
        # Write chunks in this json file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return f"Saved student_search_results to {output_path}"

    except Exception as e:
        raise Exception(
            "Error while building student search results "
            f"in 'build_student_search_results': {e}"
        )


def recover_chunks(
    retrieved_sources: list[MinimalSource],
    chunks: list[TextChunk],
) -> list[TextChunk]:
    """Recover full chunk texts from saved source locations.

    A search result stores only file path and character indexes. To generate an
    answer, we need the actual text too. This function matches each saved
    source back to the corresponding `TextChunk`, while preserving the
    retrieval order.
    """
    chunks_by_source = {
        (
            chunk.source.file_path,
            chunk.source.first_character_index,
            chunk.source.last_character_index,
        ): chunk
        for chunk in chunks
    }

    return [
        chunks_by_source[
            (
                source.file_path,
                source.first_character_index,
                source.last_character_index,
            )
        ]
        for source in retrieved_sources
        if (
            source.file_path,
            source.first_character_index,
            source.last_character_index,
        )
        in chunks_by_source
    ]


def build_student_search_results_and_answer(
    student_search_results_path: Path, save_directory: Path
) -> str:
    """Generate answers for a previously created search-results file.

    This function loads the retrieved sources, recovers their text chunks,
    builds one prompt per question, generates an answer, and saves the result
    as a `StudentSearchResultsAndAnswer` JSON file.
    """

    try:
        if not student_search_results_path.exists():
            raise Exception(
                "StudentSearchResults does not exists: search dataset first."
            )

        save_directory.mkdir(parents=True, exist_ok=True)
        output_path = save_directory / student_search_results_path.name

        search_results_with_answers: list[MinimalAnswer] = []
        # load ssr dict and convert it in StudentSearchResult object
        ssr = StudentSearchResults.model_validate(
            load_json(student_search_results_path)
        )
        # load chunks and convertir it in TextChunks object
        chunks = [
            TextChunk.model_validate(chunk)
            for chunk in load_json(state.save_chunks_path)
        ]
        for result in tqdm(ssr.search_results, desc="Generating answers"):
            # build prompt with the question and its best chunks
            prompt = build_prompt(
                recover_chunks(result.retrieved_sources, chunks),
                result.question,
            )
            search_results_with_answers.append(
                MinimalAnswer(
                    **result.model_dump(), answer=get_llm_answer(prompt)
                )
            )

        ssraa = StudentSearchResultsAndAnswer(
            search_results=search_results_with_answers, k=ssr.k
        )

        # convert ssraa to dict
        data = ssraa.model_dump()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        questions_number = len(ssr.search_results)

        return (
            f"Loaded {questions_number} questions from "
            f"{student_search_results_path}\n"
            f"Processed {questions_number} of {questions_number} questions\n"
            f"Saved student_search_results_and_answer to {output_path}"
        )

    except Exception as e:
        raise RuntimeError(
            "Error while building student search results "
            "in 'build_student_search_results'"
        ) from e
