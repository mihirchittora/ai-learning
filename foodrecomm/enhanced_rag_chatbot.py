"""Hugging Face-powered RAG chatbot for the food recommendation dataset."""

from __future__ import annotations

import os
from pathlib import Path

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from shared_functions import (
    create_similarity_search_collection,
    load_food_data,
    perform_similarity_search,
    populate_similarity_collection,
)


DATA_FILE = Path(__file__).with_name("FoodDataSet.json")
MODEL_ID = os.getenv("HF_MODEL_ID", "deepseek-ai/DeepSeek-V3-0324")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))


def get_huggingface_token() -> str:
    """Read the Hugging Face token without hardcoding or printing it."""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Hugging Face token is missing. Set HF_TOKEN in your environment."
        )
    return token


def create_huggingface_llm() -> ChatHuggingFace:
    """Create the Hugging Face chat model using the environment token."""
    endpoint = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        task="text-generation",
        huggingfacehub_api_token=get_huggingface_token(),
        max_new_tokens=512,
        temperature=0.2,
    )
    return ChatHuggingFace(llm=endpoint)


def build_context(results: list[dict]) -> str:
    """Convert retrieved food records into context for the language model."""
    if not results:
        return "No matching food records were retrieved."

    context_parts = []
    for index, result in enumerate(results, 1):
        context_parts.append(
            f"{index}. {result['food_name']}\n"
            f"   Cuisine: {result['cuisine_type']}\n"
            f"   Calories: {result['food_calories_per_serving']} per serving\n"
            f"   Description: {result['food_description']}\n"
            f"   Similarity: {result['similarity_score']:.1%}"
        )

    return "\n\n".join(context_parts)


def answer_question(question: str, collection, llm: ChatHuggingFace) -> tuple[str, list[dict]]:
    """Retrieve relevant foods and generate a grounded Hugging Face answer."""
    results = perform_similarity_search(collection, question, TOP_K)
    context = build_context(results)

    prompt = f"""You are a helpful food recommendation assistant.

Answer the user's question using only the food records in the context below.
If the context does not contain enough information, say so instead of inventing
facts. Recommend specific foods when appropriate and keep the answer concise.

Retrieved food context:
{context}

User question: {question}

Answer:"""

    response = llm.invoke(prompt)
    answer = str(getattr(response, "content", response)).strip()
    return answer, results


def print_help() -> None:
    print("\nCommands:")
    print("  help           Show this help")
    print("  quit / exit    Exit the chatbot")
    print("\nExamples:")
    print("  Recommend a light Indian dinner")
    print("  What chocolate desserts are available?")
    print("  Find something under 300 calories")


def run_chatbot(collection, llm: ChatHuggingFace) -> None:
    print("\n" + "=" * 60)
    print("🤖 HUGGING FACE FOOD RAG CHATBOT")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print("Type a food question, 'help', or 'quit'.")

    while True:
        try:
            question = input("\n🍽️  Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            return

        if not question:
            print("Please enter a question.")
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("👋 Goodbye!")
            return
        if question.lower() in {"help", "h"}:
            print_help()
            continue

        try:
            answer, sources = answer_question(question, collection, llm)
            print(f"\n💬 {answer}")

            if sources:
                source_names = ", ".join(result["food_name"] for result in sources)
                print(f"\n📚 Retrieved foods: {source_names}")
        except Exception as error:
            print(f"❌ Hugging Face request failed: {error}")


def main() -> None:
    print("🍽️  Enhanced Food RAG Chatbot")
    print("Loading food database...")

    try:
        food_items = load_food_data(DATA_FILE)
        collection = create_similarity_search_collection(
            "enhanced_food_rag",
            {"description": "Food RAG retrieval collection"},
        )
        populate_similarity_collection(collection, food_items)
        llm = create_huggingface_llm()
    except Exception as error:
        print(f"❌ Error initializing chatbot: {error}")
        return

    print(f"✅ Loaded {len(food_items)} food items")
    run_chatbot(collection, llm)


if __name__ == "__main__":
    main()
