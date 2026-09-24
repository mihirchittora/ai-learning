"""Functions for splitting and indexing LinkedIn profile data."""

import json
from typing import Any, Dict, List

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

import config
from modules.llm_interface import create_huggingface_embedding


def split_profile_data(profile_data: Dict[str, Any]) -> List:
    """Convert a profile dictionary into LlamaIndex text nodes."""

    if not profile_data:
        raise ValueError("The LinkedIn profile contains no data.")

    profile_text = json.dumps(profile_data, ensure_ascii=False, indent=2)
    document = Document(text=profile_text)
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    nodes = splitter.get_nodes_from_documents([document])
    if not nodes:
        raise ValueError("No profile text chunks were created.")
    return nodes


def create_vector_database(nodes: List) -> VectorStoreIndex:
    """Create an in-memory vector index with Hugging Face embeddings."""

    if not nodes:
        raise ValueError("Cannot create an index without profile text chunks.")
    return VectorStoreIndex(nodes, embed_model=create_huggingface_embedding())


def verify_embeddings(index: VectorStoreIndex) -> bool:
    """Return whether the index contains embedded nodes."""

    if index is None:
        return False

    vector_store = index.storage_context.vector_store
    vector_data = getattr(vector_store, "data", None)
    embedding_dict = getattr(vector_data, "embedding_dict", None)
    if embedding_dict is not None:
        return bool(embedding_dict)

    # Some vector stores do not expose their raw embedding dictionary.
    return bool(getattr(index.index_struct, "nodes_dict", {}))
