"""Minimal Hugging Face + Gradio chat example."""

from __future__ import annotations

import os

import gradio as gr
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


MODEL_ID = "deepseek-ai/DeepSeek-V3-0324"


def get_llm() -> ChatHuggingFace:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Configure it before sending a message.")

    endpoint = HuggingFaceEndpoint(
        repo_id=MODEL_ID,
        task="text-generation",
        huggingfacehub_api_token=token,
        max_new_tokens=256,
        temperature=0.2,
    )
    return ChatHuggingFace(llm=endpoint)


def generate_response(prompt_txt: str) -> str:
    if not prompt_txt or not prompt_txt.strip():
        return "Please enter a message."

    try:
        response = get_llm().invoke(prompt_txt.strip())
        return str(getattr(response, "content", response)).strip()
    except RuntimeError as exc:
        return f"Error: {exc}"
    except Exception:
        return "Error: Hugging Face inference failed. Check HF_TOKEN and try again."


chat_application = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(
        label="Input",
        lines=2,
        placeholder="Type your message here...",
    ),
    outputs=gr.Textbox(label="Output"),
    title="Hugging Face Chatbot",
    description="Ask a question using the Hugging Face model.",
)


if __name__ == "__main__":
    chat_application.launch(server_name="127.0.0.1")
