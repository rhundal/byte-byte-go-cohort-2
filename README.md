# ByteByteGo AI Engineering Bootcamp — Cohort 2

Six hands-on projects covering the full AI engineering stack: LLM foundations, RAG, agentic systems, reasoning models, multi-modal generation, and MCP/A2A communication protocols. Built during the ByteByteGo AI Engineering Cohort 2 (Nov 8 – Dec 14).


---

## Project 1 — Build an LLM Playground

Explores the foundational mechanics of large language models from the ground up. Covers the full LLM pipeline including tokenization (word-level, character-level, and Byte Pair Encoding), the Transformer architecture, and how to load and run pretrained models using HuggingFace. Implements and compares text generation strategies including greedy search, beam search, top-k, and top-p sampling. Contrasts base completion models with instruction-tuned models.

**Tech Stack**
- Language: Python
- Libraries: `torch`, `transformers`, `tiktoken`, `huggingface_hub`
- Environment: Jupyter Notebook (conda) / Google Colab

**How to Run**

Option 1 — Google Colab (recommended):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bytebyteai/ai-eng-projects-2/blob/main/project_1/lm_playground.ipynb)

Option 2 — Run locally:

```bash
# Create and activate the conda environment
conda env create -f env.yaml && conda activate llm_playground

# Register as a Jupyter kernel
python -m ipykernel install --user --name=llm_playground --display-name "llm_playground"

# Launch notebook
jupyter notebook lm_playground.ipynb
```

---

## Project 2 — Build a Customer Support Chatbot (RAG)

Builds a production-style RAG-based customer support chatbot for an imaginary e-commerce store. Covers document ingestion and chunking, embedding generation, FAISS vector indexing, retrieval-augmented generation using LangChain and Ollama, and packaging everything in a Streamlit web UI.

**Tech Stack**
- Language: Python
- Libraries: `langchain`, `langchain-community`, `faiss-cpu`, `ollama`, `streamlit`, `pypdf`, `sentence-transformers`
- Models: Gemma3-1B (via Ollama, runs locally)
- Environment: Jupyter Notebook (conda)

**How to Run**

```bash
# Pull the local model first
ollama pull gemma3:1b

# Create and activate the conda environment
conda env create -f environment.yml && conda activate rag-chatbot

# Register as a Jupyter kernel
python -m ipykernel install --user --name=rag-chatbot --display-name "rag-chatbot"

# Launch notebook
jupyter notebook rag_chatbot.ipynb
```

---

## Project 3 — Build an "Ask-the-Web" Agent (Perplexity-style)

Implements a fully agentic web research system modeled after Perplexity AI. Covers the distinction between agentic systems and standard LLMs, tool calling, function schemas, and multi-step reasoning workflows. Builds capabilities including prompt chaining, routing, and orchestrator-worker patterns using LangChain and Ollama.

**Tech Stack**
- Language: Python
- Libraries: `langchain`, `langchain-community`, `openai`, `requests`
- Models: Gemma3-1B (via Ollama, runs locally)
- Environment: Jupyter Notebook (conda)

**How to Run**

```bash
# Pull the local model first
ollama pull gemma3:1b

# Create and activate the conda environment
conda env create -f environment.yml && conda activate web_agent

# Register as a Jupyter kernel
python -m ipykernel install --user --name=web_agent --display-name "web_agent"

# Launch notebook
jupyter notebook ask_the_web_agent.ipynb
```

---

## Project 4 — Build "Deep Research" Capability

Implements advanced reasoning and inference-time scaling techniques inspired by OpenAI's o-family models and DeepSeek-R1. Covers zero-shot and few-shot Chain-of-Thought prompting, self-consistency sampling, sequential revision, Tree of Thoughts, and an introduction to STaR (Self-Taught Reasoner) for training reasoning-capable models. Culminates in a deep research agent combining step-by-step reasoning with live web search.

**Tech Stack**
- Language: Python
- Libraries: `openai`, `torch`, `transformers`
- Models: `llama3.2:3b` and `deepseek-r1:8b` (via Ollama, run locally)
- Environment: Jupyter Notebook (conda)

**How to Run**

```bash
# Pull the local models first
ollama pull llama3.2:3b
ollama pull deepseek-r1:8b

# Create and activate the conda environment
conda env create -f environment.yaml && conda activate deep_research

# Register as a Jupyter kernel
python -m ipykernel install --user --name=deep_research --display-name "deep_research"

# Launch notebook
jupyter notebook deep_research.ipynb
```

---

## Project 5 — Build a Multi-Modal Generation Agent

Builds an end-to-end multi-modal generation service using open-source diffusion models. Covers Text-to-Image generation using Stable Diffusion XL, Text-to-Video generation, and a unified multi-modal agent that routes requests to the correct modality based on user intent. Includes a Gradio UI for interactive use.

> ⚠️ **GPU required.** These models are computationally heavy and run best on a GPU. Google Colab with GPU acceleration is strongly recommended.

**Tech Stack**
- Language: Python
- Libraries: `diffusers`, `torch`, `transformers`, `huggingface_hub`, `matplotlib`, `gradio`
- Models: Stable Diffusion XL (`stabilityai/stable-diffusion-xl-base-1.0`), `damo-vilab/text-to-video-ms-1.7b`
- Environment: Google Colab (GPU)

**How to Run**

1. Open in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bytebyteai/ai-eng-projects/blob/main/project_5/multimodal_agent.ipynb)
2. Enable GPU: Runtime → Change runtime type → GPU
3. Create a HuggingFace account and generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Paste your token when prompted in the notebook


---

## About

These projects were completed as part of the [ByteByteGo AI Engineering Bootcamp](https://bytebyteai.com), a 6-week cohort-based course covering the full AI engineering stack from LLM foundations to production agentic systems.