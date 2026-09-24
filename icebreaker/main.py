"""Command-line entry point for the Icebreaker Bot."""

import argparse
import logging
from typing import Any

import config
from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import (
    create_vector_database,
    split_profile_data,
    verify_embeddings,
)
from modules.llm_interface import change_llm_model
from modules.query_engine import answer_user_query, generate_initial_facts


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def process_linkedin(linkedin_url: str, api_key: str | None = None, mock: bool = False) -> Any:
    """Build and return a vector index for a LinkedIn profile."""

    profile = extract_linkedin_profile(linkedin_url, api_key=api_key, mock=mock)
    nodes = split_profile_data(profile)
    index = create_vector_database(nodes)
    if not verify_embeddings(index):
        raise RuntimeError("The profile embeddings could not be created.")
    return index


def chatbot_interface(index: Any) -> None:
    """Run a small interactive command-line chat loop."""

    print(generate_initial_facts(index))
    print("Ask questions about the profile. Type 'exit' to stop.")
    while True:
        user_query = input("> ").strip()
        if user_query.lower() in {"exit", "quit", "bye"}:
            break
        if not user_query:
            print("Please enter a question.")
            continue
        try:
            print(answer_user_query(index, user_query))
        except (ValueError, RuntimeError) as exc:
            print(f"Error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LinkedIn Profile Icebreaker Bot")
    parser.add_argument("--url", help="LinkedIn profile URL")
    parser.add_argument("--api-key", help="Proxycurl API key for live profiles")
    parser.add_argument("--mock", action="store_true", help="Use the sample profile")
    parser.add_argument("--model", help="Hugging Face model ID")
    args = parser.parse_args()

    use_mock = args.mock or not args.url
    linkedin_url = args.url or "https://www.linkedin.com/in/example/"
    api_key = args.api_key or config.PROXYCURL_API_KEY

    if args.model:
        change_llm_model(args.model)
    if not use_mock and not api_key:
        parser.error("--api-key is required unless --mock is used")

    try:
        index = process_linkedin(linkedin_url, api_key=api_key, mock=use_mock)
        chatbot_interface(index)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
