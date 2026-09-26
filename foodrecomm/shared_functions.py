"""Shared data-loading and vector-search helpers for the food recommender."""

import json
import hashlib
import math
import re
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions



class _KeywordEmbeddingFunction:
    """Small offline fallback when the Hugging Face model is unavailable."""

    def __init__(self, dimensions=256):
        self.dimensions = dimensions

    def __call__(self, input):
        return [self._embed(text) for text in input]

    def _embed(self, text):
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude:
            vector = [value / magnitude for value in vector]

        return vector


try:
    _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        local_files_only=True,
    )
except Exception:
    print(
        "⚠️  Hugging Face model is not available locally; "
        "using offline keyword search."
    )
    _embedding_function = _KeywordEmbeddingFunction()

_client = chromadb.Client()


def load_food_data(file_path):
    """Load and validate the food dataset from a JSON file."""
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Food dataset must contain a JSON list")

    return data


def _food_document(food):
    ingredients = ", ".join(food.get("food_ingredients", []))

    return (
        f"Food: {food.get('food_name', '')}. "
        f"Description: {food.get('food_description', '')}. "
        f"Cuisine: {food.get('cuisine_type', '')}. "
        f"Ingredients: {ingredients}. "
        f"Cooking method: {food.get('cooking_method', '')}. "
        f"Health benefits: {food.get('food_health_benefits', '')}."
    )


def _food_metadata(food, index):
    return {
        "food_id": str(food.get("food_id", index + 1)),
        "food_name": str(food.get("food_name", "")),
        "food_description": str(food.get("food_description", "")),
        "food_calories_per_serving": int(
            food.get("food_calories_per_serving", 0)
        ),
        "cuisine_type": str(food.get("cuisine_type", "")),
    }


def create_similarity_search_collection(name, metadata=None):
    """Create or retrieve the Chroma collection used for food search."""
    collection_metadata = {"hnsw:space": "cosine"}
    if metadata:
        collection_metadata.update(metadata)

    return _client.get_or_create_collection(
        name=name,
        metadata=collection_metadata,
        embedding_function=_embedding_function,
    )


def populate_similarity_collection(collection, food_items):
    """Index food descriptions and their display metadata."""
    documents = [_food_document(food) for food in food_items]
    metadatas = [
        _food_metadata(food, index)
        for index, food in enumerate(food_items)
    ]
    # Use the row number so duplicate dataset food_id values remain distinct.
    ids = [f"food_{index + 1}" for index, _ in enumerate(food_items)]

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )


def perform_similarity_search(collection, query, limit=5):
    """Return the closest foods for a natural-language query."""
    item_count = collection.count()
    if item_count == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(limit, item_count),
        include=["metadatas", "distances"],
    )

    return _format_search_results(results)


def perform_filtered_similarity_search(
    collection,
    query,
    cuisine_filter=None,
    max_calories=None,
    n_results=5,
):
    """Search by similarity while optionally filtering cuisine and calories."""
    item_count = collection.count()
    if item_count == 0:
        return []

    conditions = []
    if cuisine_filter:
        conditions.append({"cuisine_type": {"$eq": str(cuisine_filter)}})
    if max_calories is not None:
        conditions.append({
            "food_calories_per_serving": {"$lte": int(max_calories)},
        })

    if len(conditions) == 1:
        where = conditions[0]
    elif conditions:
        where = {"$and": conditions}
    else:
        where = None

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, item_count),
        where=where,
        include=["metadatas", "distances"],
    )

    return _format_search_results(results)


def _format_search_results(results):
    """Convert Chroma's nested query response into application records."""
    output = []
    for metadata, distance in zip(
        results["metadatas"][0],
        results["distances"][0],
    ):
        similarity_score = max(0.0, min(1.0, 1.0 - float(distance)))
        output.append({**metadata, "similarity_score": similarity_score})

    return output
