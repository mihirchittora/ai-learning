# LinkedIn Icebreaker Bot

An educational Gradio application that extracts LinkedIn profile data, indexes it with LlamaIndex and Hugging Face embeddings, and generates personalized icebreakers with a Hugging Face model.

## Architecture

1. Load a profile from the sample dataset or Proxycurl.
2. Serialize and split the profile into LlamaIndex nodes.
3. Embed nodes locally with `sentence-transformers/all-MiniLM-L6-v2`.
4. Retrieve relevant nodes from an in-memory LlamaIndex vector index.
5. Generate answers with `deepseek-ai/DeepSeek-V3-0324` through Hugging Face.
6. Display the result in Gradio.

## Setup

Python 3.11 is recommended.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set the Hugging Face token in the environment. Do not put it in source code:

```bash
export HF_TOKEN="your_hugging_face_token"
```

The token is used only for Hugging Face inference. The sample profile does not require a Proxycurl key.

## Run the Gradio app

```bash
python app.py
```

Open `http://127.0.0.1:7860`.

The default UI uses the sample profile. For a live LinkedIn profile, uncheck **Use Mock Data** and provide a Proxycurl API key.

## Run the command-line version

```bash
python main.py --mock
```

## Project structure

```text
config.py                    Model and prompt settings
app.py                       Gradio application
main.py                      Command-line application
modules/data_extraction.py   Sample/Proxycurl profile loading
modules/data_processing.py   Splitting, embedding, and indexing
modules/llm_interface.py     Hugging Face embedding and LLM adapters
modules/query_engine.py     Retrieval and question answering
```
