"""Pydantic data models used by the RAG pipeline."""

from typing import List, Union
from pydantic import Field, BaseModel
from uuid import uuid4


class MinimalSource(BaseModel):
    """Minimal source location inside a repository file.
    (Represent a part of the file, a chunk)"""

    file_path: str
    first_character_index: int
    last_character_index: int


class TextChunk(BaseModel):
    """The content of the MinimalSource"""

    source: MinimalSource
    text: str


class UnansweredQuestion(BaseModel):
    """Question without a reference answer."""

    # default_factory: créer automatiquement un question_id à chaque
    # nouvel objet si jamais il n’y en a pas. uuid4() génère un id unique
    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question with a reference answer and expected sources."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset containing answered or unanswered RAG questions."""

    # Dans rag_questions, il peut y avoir des AnsweredQuestion ou
    # des UnansweredQuestion, d'où le union
    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Search results for one question."""

    question_id: str
    question: str
    question_str: str = ""
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Generated answer with retrieved sources."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results file produced by the student pipeline."""

    # c'est ce que la moulinette va lire
    search_results: List[MinimalSearchResults]
    # combien de sources max j'ai demandé par question
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Answer file produced by the student pipeline."""

    # MinimalAnswer ajoute le champ answer. En gros c'est comme
    # StudentSearchResult mais on rajoute la reponse dans chaque
    # element de search_results.
    search_results: List[MinimalAnswer]
    k: int
