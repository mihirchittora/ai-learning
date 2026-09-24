"""PDF RAG chatbot using Hugging Face, Chroma, LangChain, and Gradio."""

from __future__ import annotations

import os
import logging
from typing import Any, List, Optional

import gradio as gr
from chromadb.config import Settings
from huggingface_hub import InferenceClient, get_token as get_huggingface_token
from langchain_core.language_models.llms import LLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Chroma 0.4.24 can emit an incompatible posthog callback warning.
logging.getLogger("chromadb.telemetry.product.posthog").disabled = True


# ============================================================
# Hugging Face LLM
# ============================================================

class HuggingFaceRouterLLM(LLM):

    model: str = "deepseek-ai/DeepSeek-V3-0324"
    max_tokens: int = 256
    temperature: float = 0.5

    @property
    def _llm_type(self) -> str:
        return "huggingface-router"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:

        # Prefer HF_TOKEN. A cached Hugging Face login is a safe local fallback
        # when the token was configured with `hf auth login` instead of exported.
        token = os.getenv("HF_TOKEN") or get_huggingface_token()

        if not token:
            raise RuntimeError(
                "HF_TOKEN is not set and no cached Hugging Face login was found. "
                "Export HF_TOKEN or run `hf auth login`."
            )

        try:
            client_args = {"model": self.model, "token": token}
            try:
                client = InferenceClient(provider="auto", **client_args)
            except TypeError:
                # Older huggingface_hub versions do not accept `provider`.
                client = InferenceClient(**client_args)

            if hasattr(client, "chat") and hasattr(client.chat, "completions"):
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                answer = response.choices[0].message.content
            else:
                answer = client.text_generation(
                    prompt,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    return_full_text=False,
                )

            if not answer:
                raise RuntimeError("Hugging Face returned an empty response.")

            return str(answer)

        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Hugging Face inference failed. Check the token, model, and network."
            ) from exc


def get_llm():
    return HuggingFaceRouterLLM(
        model="deepseek-ai/DeepSeek-V3-0324",
        max_tokens=256,
        temperature=0.5
    )


# ============================================================
# Embedding model
# ============================================================

def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# Document loader
# ============================================================

def document_loader(file):

    if file is None:
        raise ValueError("Please upload a PDF file.")

    # Gradio type="filepath" gives us a string path
    loader = PyPDFLoader(file)

    documents = [document for document in loader.load() if document.page_content.strip()]

    if not documents:
        raise ValueError(
            "The PDF contains no extractable text."
        )

    return documents


# ============================================================
# Text splitter
# ============================================================

def text_splitter(data):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    chunks = splitter.split_documents(data)

    if not chunks:
        raise ValueError(
            "No text chunks were created."
        )

    return chunks


# ============================================================
# Vector database
# ============================================================

def vector_database(chunks):

    embedding_model = get_embedding_model()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        client_settings=Settings(anonymized_telemetry=False),
    )

    return vectordb


# ============================================================
# Retriever
# ============================================================

def get_retriever(vectordb):

    return vectordb.as_retriever(
        search_kwargs={"k": 4}
    )


# ============================================================
# RAG
# ============================================================

def retriever_qa(file, query):

    if file is None:
        return "Please upload a PDF."

    if not query or not query.strip():
        return "Please enter a question."

    try:

        # 1. Load PDF
        documents = document_loader(file)

        # 2. Split documents
        chunks = text_splitter(documents)

        # 3. Create vector database
        vectordb = vector_database(chunks)

        # 4. Create retriever
        retriever_instance = get_retriever(vectordb)

        # 5. Get LLM
        llm = get_llm()

        # 6. Create RAG chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_instance,
            return_source_documents=False
        )

        # 7. Ask question
        response = qa.invoke({
            "query": query
        })

        return response["result"]

    except (ValueError, RuntimeError) as exc:
        return f"Error: {exc}"
    except Exception:
        return "Error: The document could not be processed. Please try again."


# ============================================================
# Gradio UI
# ============================================================

demo = gr.Interface(
    fn=retriever_qa,

    inputs=[
        gr.File(
            label="Upload PDF",
            file_types=[".pdf"],
            type="filepath"
        ),

        gr.Textbox(
            label="Ask a question",
            lines=3,
            placeholder="Ask something about the PDF..."
        )
    ],

    outputs=gr.Textbox(
        label="Answer",
        lines=10
    ),

    title="Hugging Face RAG Chatbot",

    description=(
        "Upload a PDF and ask questions about its contents."
    )
)


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False
    )
