<h1 align="center">RAG Against the Machine</h1>

<p align="center">
  <strong>Retrieval-Augmented Generation system for codebases using BM25 indexing and local LLM generation.</strong><br>
  <em>Created as part of the 42 curriculum by bpasquer.</em>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#retrieval-method">Retrieval</a> •
  <a href="#answer-generation">Generation</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#example-usage">Usage</a> •
  <a href="#license">License</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/uv-Package%20Manager-DE5FE9?style=flat-square&logo=astral&logoColor=white" alt="uv" />
  <img src="https://img.shields.io/badge/BM25s-Lexical%20Search-FF6F00?style=flat-square" alt="BM25s" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/Qwen-Qwen3--0.6B-6366F1?style=flat-square" alt="Qwen" />
  <img src="https://img.shields.io/badge/Pydantic-v2-E92063?style=flat-square&logo=pydantic&logoColor=white" alt="Pydantic" />
  <img src="https://img.shields.io/badge/License-MIT-black?style=flat-square" alt="License" />
</p>

---

## Overview

This project is a small Retrieval-Augmented Generation system, often called a
RAG. Its goal is to answer questions about the vLLM codebase by first searching
inside the project files, then giving the most relevant pieces of text to a
language model.

A normal language model answers mostly from what it learned during training. A
RAG system adds an external knowledge base. In this project, the knowledge base
is built from the provided vLLM repository. When a user asks a question, the
program retrieves relevant code or documentation chunks and can use them as
context for Qwen/Qwen3-0.6B.

The project has four main parts:

- indexing the vLLM repository;
- retrieving relevant chunks with BM25;
- generating answers with Qwen/Qwen3-0.6B;
- evaluating retrieval quality with recall@k.

## Architecture

The pipeline is organized as follows:

```text
vLLM repository
      |
      v
File collection (.py and .md files)
      |
      v
Chunking
      |
      v
BM25 indexing
      |
      v
Saved chunks + saved BM25 index
      |
      v
Search / Evaluation / Answer generation
```

The important source files are:

- `src/chunking.py`: reads files and creates text chunks;
- `src/indexing.py`: builds the BM25 corpus and saves the index;
- `src/retrieval.py`: loads the index and returns the best chunks for a query;
- `src/generation.py`: builds prompts and loads Qwen only when needed;
- `src/evaluation.py`: computes recall@k;
- `src/build_results.py`: creates JSON files for dataset search and answers;
- `src/cli.py`: exposes the Python Fire command-line interface.

Generated files are stored under `data/processed/` and `data/output/`.

## Installation

This project uses Python 3.10 or later and `uv` as the package manager.

Install the project dependencies:

```bash
make install
```

Index the vLLM repository:

```bash
uv run python -m src index --max_chunk_size 2000
```

The indexing command creates:

```text
data/processed/chunks/chunks.json
data/processed/bm25_index/
```

Search for a single question:

```bash
uv run python -m student search "How to configure the OpenAI server?" --k 10
```

Search a full dataset:

```bash
uv run python -m student search_dataset \
  --dataset_path datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results \
  --k 10
```

This writes:

```text
data/output/search_results/dataset_docs_public.json
```

Evaluate search results:

```bash
uv run python -m student evaluate \
  --student_answer_path data/output/search_results/dataset_docs_public.json \
  --dataset_path datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
  --k 10 \
  --max_context_length 2000
```

Answer a single question with the language model:

```bash
uv run python -m student answer "How to configure the OpenAI server?" --k 10
```

Generate answers from a search result file:

```bash
uv run python -m student answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer
```

Run linting and type checking:

```bash
make lint
```

Clean generated files:

```bash
make clean
```

## Chunking Strategy

The system indexes Python and Markdown files from the vLLM repository. Each file
is read as text and split into chunks with a configurable maximum size. The
maximum allowed chunk size is 2000 characters, as required by the subject.

Chunks overlap slightly. This helps avoid losing useful information when an
important sentence or code block is located at the boundary between two chunks.
For example, if a function description starts near the end of one chunk, the
overlap gives the next chunk a chance to contain enough context too.

Each chunk stores:

- the source file path;
- the first character index in the original file;
- the last character index in the original file;
- the chunk text.

These character positions are important because the evaluator checks whether
retrieved sources overlap with the expected sources.

## Retrieval Method

The retrieval system uses BM25 through the `bm25s` package.

BM25 is a ranking algorithm used by search engines. It compares a query with a
set of documents and gives higher scores to documents that contain useful query
terms. It is stronger than a simple word count because it also considers how
common or rare a term is in the whole corpus.

The project uses BM25 in this order:

1. Build a corpus from all chunks.
2. Tokenize the corpus.
3. Build and save the BM25 index.
4. Tokenize each user query.
5. Retrieve the top-k chunk indexes.
6. Convert those indexes back to `TextChunk` objects.

The returned sources follow the required model:

```json
{
  "file_path": "docs/example.md",
  "first_character_index": 120,
  "last_character_index": 900
}
```

## Answer Generation

The answer generation step uses `Qwen/Qwen3-0.6B`.

The model is loaded lazily, which means it is loaded only when an answer command
needs it. This keeps search, indexing, evaluation, and CLI startup fast. Without
lazy loading, even a simple search command would load the model unnecessarily.

For answer generation, the program:

1. retrieves the top-k chunks for the question;
2. builds a prompt containing only those chunks as context;
3. asks the model to answer using only the provided context;
4. returns the generated text.

The prompt asks the model to say that it does not know when the answer is not
present in the retrieved context. This reduces hallucinations.

## Evaluation

The retrieval system is evaluated with recall@k.

Recall@k answers this question:

```text
Among the correct sources, how many were found in the first k retrieved sources?
```

Example:

```text
Expected sources: A, B
Retrieved top 5:  C, A, D, E, F
```

Only source A was found, so:

```text
recall@5 = 1 / 2 = 0.5
```

The subject considers a source found when the retrieved source overlaps with an
expected source by at least 5%. The comparison uses the file path and the
character ranges.

The evaluation command reports recall@1, recall@3, recall@5, and recall@k.

## Performance Analysis

Public dataset results after indexing with `max_chunk_size = 2000`:

```text
Documentation questions:
recall@1:  0.64
recall@3:  0.76
recall@5:  0.85
recall@10: 0.88

Code questions:
recall@1:  0.47
recall@3:  0.63
recall@5:  0.66
recall@10: 0.72
```

The required thresholds are:

- docs recall@5: at least 0.80;
- code recall@5: at least 0.50.

On the public datasets, this implementation reaches both thresholds.

Indexing was also tested locally and completed in a few seconds on the provided
vLLM repository, which is below the 5 minute limit required by the subject.

## Design Decisions

BM25 was chosen because it is simple, fast, and reliable for a first retrieval
baseline. It also satisfies the subject requirement to implement either TF-IDF
or BM25.

The project stores both the chunks and the BM25 index on disk. This avoids
re-indexing the repository before every query.

The output files use Pydantic models so the generated JSON follows the expected
structure. This is useful because the evaluator depends on exact field names
such as `question_id`, `retrieved_sources`, `file_path`,
`first_character_index`, and `last_character_index`.

The model is loaded only inside the generation path. Retrieval and evaluation do
not need the language model, so they should not pay its startup cost.

## Challenges Faced

One challenge was understanding that retrieval quality must be measured before
improving generation. If the retrieved chunks are wrong, the language model will
not have the right context and cannot answer reliably.

Another challenge was the recall@k metric. The expected source and retrieved
source do not have to be exactly identical. They only need enough character
overlap in the same file. Implementing this correctly required comparing
intervals.

The command-line interface also needed careful path handling. The subject gives
a save directory, not a full output file path. The program therefore creates the
directory and writes a JSON file inside it using the input dataset name.

Finally, loading Qwen directly at import time made every command slow. The model
is now loaded lazily so that search and evaluation remain fast.

## Example Usage

Full retrieval evaluation workflow:

```bash
make install

uv run python -m student index --max_chunk_size 2000

uv run python -m student search_dataset \
  --dataset_path datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results \
  --k 10

uv run python -m student evaluate \
  --student_answer_path data/output/search_results/dataset_docs_public.json \
  --dataset_path datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
  --k 10 \
  --max_context_length 2000
```

Code dataset workflow:

```bash
uv run python -m student search_dataset \
  --dataset_path datasets_public/public/UnansweredQuestions/dataset_code_public.json \
  --save_directory data/output/search_results \
  --k 10

uv run python -m student evaluate \
  --student_answer_path data/output/search_results/dataset_code_public.json \
  --dataset_path datasets_public/public/AnsweredQuestions/dataset_code_public.json \
  --k 10 \
  --max_context_length 2000
```

Single question workflow:

```bash
uv run python -m student search "What is prefix caching in vLLM?" --k 10

uv run python -m student answer "What is prefix caching in vLLM?" --k 10
```

## Resources

Useful resources for this project:

- Python documentation: https://docs.python.org/3/
- Pydantic documentation: https://docs.pydantic.dev/
- Python Fire documentation: https://github.com/google/python-fire
- uv documentation: https://docs.astral.sh/uv/
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers
- Qwen model page: https://huggingface.co/Qwen/Qwen3-0.6B
- BM25 overview: https://en.wikipedia.org/wiki/Okapi_BM25
- bm25s package: https://github.com/xhluca/bm25s
- vLLM documentation: https://docs.vllm.ai/

AI was used as a learning and debugging assistant during the project. It helped
explain RAG concepts, recall@k, path handling, lazy model loading, and linting
errors. The implementation choices and final code were reviewed and tested
locally so the project remains understandable and maintainable by its author.

---

## License

This project is licensed under the [MIT License](LICENSE). Educational project created as part of the 42 curriculum.
