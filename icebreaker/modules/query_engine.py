"""Query functions for the profile vector index."""

from typing import Any

from llama_index.core import PromptTemplate, VectorStoreIndex

import config
from modules.llm_interface import create_huggingface_llm


def _query(index: VectorStoreIndex, template: str, query: str) -> str:
    if index is None:
        raise ValueError("No profile has been processed yet.")

    query_engine = index.as_query_engine(
        llm=create_huggingface_llm(),
        similarity_top_k=config.SIMILARITY_TOP_K,
        text_qa_template=PromptTemplate(template),
    )
    try:
        response = query_engine.query(query)
    except Exception as exc:
        raise RuntimeError(
            "The profile could not be retrieved or Hugging Face generation failed."
        ) from exc
    return str(response).strip()


def generate_initial_facts(index: VectorStoreIndex) -> str:
    """Generate three profile facts using retrieved profile context."""

    return _query(index, config.INITIAL_FACTS_TEMPLATE, "Generate the facts.")


def answer_user_query(index: VectorStoreIndex, user_query: str) -> Any:
    """Answer a question using the indexed LinkedIn profile."""

    if not user_query or not user_query.strip():
        raise ValueError("Please enter a question.")
    return _query(index, config.USER_QUESTION_TEMPLATE, user_query.strip())
