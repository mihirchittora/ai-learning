"""Gradio web interface for the Hugging Face Icebreaker Bot."""

import logging
import os
import socket
import sys
import uuid
from typing import Any, Dict, List, Optional

import gradio as gr

import config
from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import (
    create_vector_database,
    split_profile_data,
    verify_embeddings,
)
from modules.llm_interface import change_llm_model
from modules.query_engine import answer_user_query, generate_initial_facts


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)

# The index is kept in memory for the current local process.
active_indices: Dict[str, Any] = {}


def _get_server_port() -> int:
    """Use the requested port or the first available local Gradio port."""

    configured_port = os.getenv("GRADIO_SERVER_PORT")
    if configured_port:
        return int(configured_port)

    for port in range(7860, 7870):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port

    raise OSError("No available port found in the range 7860-7869.")


def process_profile(
    linkedin_url: str,
    api_key: str,
    use_mock: bool,
    selected_model: str,
) -> tuple[str, Optional[str]]:
    """Extract, embed, index, and summarize a LinkedIn profile."""

    if not use_mock and not linkedin_url:
        return "Error: enter a LinkedIn profile URL or select Use Mock Data.", None
    if not use_mock and not api_key:
        return "Error: enter a Proxycurl API key for a live profile.", None

    try:
        if selected_model and selected_model != config.LLM_MODEL_ID:
            change_llm_model(selected_model)

        profile = extract_linkedin_profile(
            linkedin_profile_url=linkedin_url,
            api_key=api_key,
            mock=use_mock,
        )
        nodes = split_profile_data(profile)
        index = create_vector_database(nodes)
        if not verify_embeddings(index):
            raise RuntimeError("The profile embeddings could not be created.")

        facts = generate_initial_facts(index)
        session_id = str(uuid.uuid4())
        active_indices[session_id] = index
        return facts, session_id
    except (ValueError, RuntimeError) as exc:
        return f"Error: {exc}", None
    except Exception:
        logger.exception("Unexpected profile processing failure")
        return "Error: the profile could not be processed. Check the terminal for details.", None


def chat_with_profile(
    session_id: Optional[str],
    user_query: str,
    chat_history: Optional[List[List[str]]],
) -> List[List[str]]:
    """Answer a question using the profile index stored for this session."""

    history = chat_history or []
    if not session_id or session_id not in active_indices:
        return history + [[user_query or "", "Error: process a profile first."]]
    if not user_query or not user_query.strip():
        return history + [["", "Error: please enter a question."]]

    try:
        answer = answer_user_query(active_indices[session_id], user_query)
    except (ValueError, RuntimeError) as exc:
        answer = f"Error: {exc}"
    except Exception:
        logger.exception("Unexpected chat failure")
        answer = "Error: the question could not be answered. Check the terminal for details."
    return history + [[user_query, answer]]


def create_gradio_interface() -> gr.Blocks:
    """Create the Gradio interface."""

    available_models = [config.LLM_MODEL_ID]
    with gr.Blocks(title="LinkedIn Icebreaker Bot") as demo:
        gr.Markdown("# LinkedIn Icebreaker Bot")
        gr.Markdown(
            "Generate personalized icebreakers from a LinkedIn profile using "
            "Hugging Face. Set `HF_TOKEN` in the environment before starting."
        )

        with gr.Tab("Process LinkedIn Profile"):
            with gr.Row():
                with gr.Column():
                    linkedin_url = gr.Textbox(
                        label="LinkedIn Profile URL",
                        placeholder="https://www.linkedin.com/in/username/",
                    )
                    api_key = gr.Textbox(
                        label="Proxycurl API Key (only for live profiles)",
                        placeholder="Your Proxycurl API key",
                        type="password",
                    )
                    use_mock = gr.Checkbox(label="Use Mock Data", value=True)
                    model_dropdown = gr.Dropdown(
                        choices=available_models,
                        label="Hugging Face LLM",
                        value=config.LLM_MODEL_ID,
                    )
                    process_btn = gr.Button("Process Profile")

                with gr.Column():
                    result_text = gr.Textbox(label="Initial Facts", lines=10)
                    session_id = gr.Textbox(label="Session ID", visible=False)

            process_btn.click(
                fn=process_profile,
                inputs=[linkedin_url, api_key, use_mock, model_dropdown],
                outputs=[result_text, session_id],
            )

        with gr.Tab("Chat"):
            gr.Markdown("Chat with the processed LinkedIn profile")
            chatbot = gr.Chatbot(height=500)
            chat_input = gr.Textbox(
                label="Ask a question about the profile",
                placeholder="What is this person's current job title?",
            )
            chat_btn = gr.Button("Send")

            for event in (chat_btn.click, chat_input.submit):
                event(
                    fn=chat_with_profile,
                    inputs=[session_id, chat_input, chatbot],
                    outputs=[chatbot],
                )

    return demo


if __name__ == "__main__":
    create_gradio_interface().launch(
        server_name="127.0.0.1",
        server_port=_get_server_port(),
        share=False,
    )
