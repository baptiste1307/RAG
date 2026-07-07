import re
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .chunking import TextChunk

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_CONTEXT_CHUNKS = 6
MAX_CONTEXT_CHARACTERS = 8000
MAX_CHUNK_CHARACTERS = 1600
MAX_PROMPT_TOKENS = 3000

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "used",
    "using",
    "what",
    "when",
    "where",
    "which",
    "with",
}

tokenizer = None

model = None


# Load the ONLY when get_llm_answer is called (performance)
def load_model() -> Any:
    """Load Qwen once and reuse it inside the same Python process.

    Loading a language model is expensive. This function keeps the tokenizer and
    model in global variables after the first load, so `answer_dataset` does
    not reload Qwen for every question.
    """
    global tokenizer, model

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            # adapte le model selon si CPU/GPU dispo, RAM limitee etc
            # ==> ou vit le modele (necessite install accelerate)
            device_map="auto",
            # Un modèle comme Qwen 0.6B :
            # en float32 → ~2.4 GB RAM
            # en float16 → ~1.2 GB RAM
            # donc 2x moins de mémoire + plus rapide sur GPU
            # ==> comment le modele est stocke
            dtype=(
                torch.float16 if torch.cuda.is_available() else torch.float32
            ),
        )

    return tokenizer, model


def _keywords(text: str) -> set[str]:
    """Extract simple search words from a sentence or chunk.

    Common words like "the", "what", and "with" are ignored because they do not
    help us decide whether a chunk is useful. The remaining words are used to
    compare a question with candidate context chunks.
    """
    return {
        word
        for word in re.findall(r"[a-zA-Z0-9_/-]+", text.lower())
        if len(word) > 2 and word not in STOP_WORDS
    }


def _chunk_generation_score(chunk: TextChunk, question_words: set[str]) -> int:
    """Count how many question keywords appear in a chunk.

    A bigger score means the chunk shares more important words with the
    question, so it is probably more useful for answer generation.
    """
    chunk_words = _keywords(chunk.text + " " + chunk.source.file_path)
    return len(question_words & chunk_words)


def _rank_chunks_for_generation(
    best_chunks: list[TextChunk], question: str
) -> list[TextChunk]:
    """Reorder retrieved chunks before building the LLM prompt.

    BM25 already returns good chunks, but small language models can be confused
    by too much context. This function puts the most question-like chunks first
    and keeps only a small number of them to reduce noise.
    """
    question_words = _keywords(question)
    ranked_chunks = sorted(
        enumerate(best_chunks),
        key=lambda item: (
            _chunk_generation_score(item[1], question_words),
            -item[0],
        ),
        reverse=True,
    )
    return [chunk for _, chunk in ranked_chunks[:MAX_CONTEXT_CHUNKS]]


def _format_limited_context(ranked_chunks: list[TextChunk]) -> str:
    """Build a context string that stays under the prompt size budget.

    Each retrieved chunk can be long, and several long chunks can make the
    prompt too large for a small local model. This helper adds sources one by
    one, trims very long chunks, and stops before the context passes the fixed
    character budget. The tokenizer also applies `MAX_PROMPT_TOKENS` later as a
    final token-level safety limit.
    """
    context_parts = []
    used_characters = 0

    for index, chunk in enumerate(ranked_chunks, 1):
        chunk_text = chunk.text[:MAX_CHUNK_CHARACTERS].strip()
        source_text = f"Source {index}: {chunk.source.file_path}\n{chunk_text}"
        if used_characters + len(source_text) > MAX_CONTEXT_CHARACTERS:
            remaining = MAX_CONTEXT_CHARACTERS - used_characters
            if remaining <= 0:
                break
            context_parts.append(source_text[:remaining].rstrip())
            break
        context_parts.append(source_text)
        used_characters += len(source_text) + 2

    return "\n\n".join(context_parts)


def build_prompt(best_chunks: list[TextChunk], question: str) -> str:
    """Build the text prompt sent to the answer system.

    The prompt contains the best context chunks, the user question, and strict
    instructions: answer only from the context, stay concise, and avoid
    repeating the question.
    """

    ranked_chunks = _rank_chunks_for_generation(best_chunks, question)
    context = _format_limited_context(ranked_chunks)

    # multi-lines f-string
    return f"""
        You are a helpful assistant.

        Use ONLY the context below to answer the question.
        If the answer cannot be found in the context, say "I don't know".
        Write only the final answer, in one concise paragraph.
        Do not explain your reasoning.
        Do not repeat the question.
        Prefer exact names, endpoints, commands, methods, and values from the
        context.

        Context:
        {context}

        Question:
        {question}

        Answer:
    """


def get_llm_answer(prompt: str) -> str:
    """Return a final answer from a prompt.

    This function loads Qwen, generates a short answer from the prompt, and
    removes common repeated sections from the output.
    """
    tokenizer, model = load_model()

    # tokenize prompt
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
        # to send all tokens on the device used by our llm model
        # otherwise they can be spreaded
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=70,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
        # For more determinists and less creative responses
        do_sample=False,
        # if llm generates EndOfSequence token, stop here
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    prompt_tokens = inputs["input_ids"]

    # p_t.shape = (x, y) (x sequence(s), y token(s))
    # => shape[1] = prompt token number (prompt length)
    tokens_to_skip = prompt_tokens.shape[1]

    # Skip prompt tokens to only keep llm answer
    # (outputs[0] = tensor containing tokenized prompt and answer)
    generated_tokens = outputs[0][tokens_to_skip:]

    # decode answer
    answer = str(
        tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )
    ).strip()

    stop_markers = [
        "\nWait,",
        "\nOkay,",
        "\nLet me",
        "\nThe answer is:",
        "\nthe answer is:",
        "\nThe answer is",
        "\nthe answer is",
        "\nAnswer:",
        "\nQuestion:",
    ]

    for marker in stop_markers:
        if marker in answer:
            answer = answer.split(marker)[0].strip()

    for prefix in ("Answer:", "Final answer:"):
        if answer.startswith(prefix):
            answer = answer.removeprefix(prefix).strip()

    boxed_match = re.search(r"\\boxed\{([^{}]+)\}", answer)
    if boxed_match:
        return boxed_match.group(1).strip()

    return answer
