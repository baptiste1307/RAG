import bm25s
from .config import global_state as state, load_json
from .chunking import create_and_save_chunks
from .models import TextChunk


def build_corpus(chunks: list[TextChunk]) -> list[str]:
    """Extract text and source file
    from each chunk"""
    return [chunk.source.file_path + "\n" + chunk.text for chunk in chunks]


def tokenize_and_save_corpus(corpus: list[str]) -> None:
    """Tokenize the corpus, createss a BM25 search engine then
    stores the index created in index_dir"""
    state.save_index_path.mkdir(parents=True, exist_ok=True)
    # tokenize corpus
    corpus_tokens = bm25s.tokenize(corpus)
    # build index (retriever = system that finds relevant documents)
    # "bm25s.BM25()" -> creates a BM25 search engine
    retriever = bm25s.BM25()
    # ".index()" ==> build inverted index: instead of 'doc -> words',
    # BM25 builds 'word -> docs containing this word'
    # ==> really fast search
    retriever.index(corpus_tokens)
    # save index -> saves BM25 index, vocabulary, stats etc.
    # ==> dont need to recalculate the entire BM25 at each launch
    retriever.save(state.save_index_path)


def index_repo(max_chunk_size: int) -> str:
    """Orchestrator function to chunk then index repo
    and save both of them in output directory"""

    if not state.save_chunks_path.exists():
        chunks = create_and_save_chunks(max_chunk_size)
    else:
        chunks = [
            TextChunk.model_validate(chunk)
            for chunk in load_json(state.save_chunks_path)
        ]

    if not state.save_index_path.exists():
        tokenize_and_save_corpus(build_corpus(chunks))
        return (
            f"Ingestion complete! Indices saved under {state.save_index_path}"
        )

    return f"Index already available under {state.save_index_path}"
