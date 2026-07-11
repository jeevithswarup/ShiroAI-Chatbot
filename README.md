<div align="center">

<img src="https://capsule-render.vercel.app/api?type=rect&color=0:0a0e1a,50:0d1a2e,100:0a0e1a&height=180&section=header&text=Local%20AI%20Chatbot%20%2B%20RAG&fontSize=46&fontColor=E8EAF0&fontAlignY=42&desc=FastAPI%20%C2%B7%20Ollama%20%C2%B7%20Qwen3%3A4B%20%C2%B7%20ChromaDB%20%C2%B7%20RAG&descAlignY=65&descColor=64FFDA&animation=fadeIn" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-0a0e1a?style=for-the-badge&logoColor=64FFDA)](https://ollama.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-E8EAF0?style=for-the-badge)](https://trychroma.com)
[![Status](https://img.shields.io/badge/RAG-In%20Progress-FFA500?style=for-the-badge)](#)

**A fully local AI chatbot built from scratch — no cloud APIs, no subscriptions, no data leaving your machine.**
Runs on your hardware using Ollama + Qwen3:4B, with streaming responses, chat history, and an evolving RAG pipeline.

</div>

---

## 📌 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [Project Structure](#-project-structure)
- [Phase 1 — Basic Chatbot](#-phase-1--basic-chatbot)
- [FastAPI](#-fastapi)
- [Pydantic Schemas](#-pydantic-schemas)
- [Separation of Concerns](#-separation-of-concerns-mainpy--servicespy)
- [Ollama Integration](#-ollama-integration)
- [Chat Message Format](#-chat-message-format)
- [System Prompt](#-system-prompt)
- [Chat History](#-chat-history)
- [Streaming Responses](#-streaming-responses)
- [Generator Functions](#-generator-functions)
- [Phase 2 — RAG Pipeline](#-phase-2--rag-pipeline)
- [Fine-Tuning vs RAG](#-fine-tuning-vs-rag)
- [RAG Architecture](#-rag-architecture)
- [PDF Reader](#-pdf-reader)
- [Current Progress](#-current-progress)
- [Roadmap](#-roadmap)

---

## 🎯 What This Project Does

This is a **local AI chatbot** that runs entirely on your own machine:

- No OpenAI API key required
- No data sent to any cloud server
- Fully private — ideal for sensitive documents
- Streaming responses so you see output token-by-token
- Memory of the conversation via chat history
- RAG (Retrieval-Augmented Generation) layer to answer questions from your own documents

---

## 📁 Project Structure

```
project/
│
├── app/
│   ├── main.py          ← FastAPI app, defines all API routes
│   ├── services.py      ← Business logic, talks to Ollama
│   ├── schemas.py       ← Pydantic request/response models
│   ├── rag.py           ← PDF reading and RAG orchestration
│   ├── vector_db.py     ← ChromaDB setup and querying
│   └── embeddings.py    ← Sentence Transformer embedding logic
│
├── documents/           ← Your PDF/text files go here
└── chroma_db/           ← ChromaDB stores vector data here
```

| File | Responsibility |
|---|---|
| `main.py` | Receives HTTP requests, sends HTTP responses |
| `services.py` | Talks to Ollama, manages chat history, handles streaming |
| `schemas.py` | Defines the shape of request/response data |
| `rag.py` | Reads PDFs, orchestrates retrieval for RAG |
| `vector_db.py` | Stores and searches document embeddings in ChromaDB |
| `embeddings.py` | Converts text chunks into numerical vectors |

---

## 🔵 Phase 1 — Basic Chatbot

### Goal

Build a REST API that takes a user's message and returns an AI-generated response from a local model.

### High-Level Architecture

```
User
 │
 │  POST /chat  {"message": "Hello"}
 ▼
FastAPI  (main.py)
 │
 │  calls ask_llm(message)
 ▼
services.py
 │
 │  ollama.chat(model, messages)
 ▼
Ollama (local inference server)
 │
 ▼
Qwen3:4B Model
 │
 ▼
Response text returned
 │
 ▼
{"response": "Hello! How can I help?"}
```

---

## ⚡ FastAPI

### What Is FastAPI?

FastAPI is a modern Python web framework for building APIs. You define routes (URLs) as Python functions, and it handles everything else — parsing requests, validating data, generating documentation, and converting responses to JSON automatically.

### Why FastAPI Instead of Flask or Django?

| Feature | Flask | Django | FastAPI |
|---|---|---|---|
| Speed | Moderate | Moderate | Very fast |
| Automatic validation | ❌ Manual | ❌ Manual | ✅ Built-in (Pydantic) |
| Auto API docs (Swagger) | ❌ Plugin needed | ❌ Plugin needed | ✅ Built-in |
| Type hints support | ❌ Optional | ❌ Optional | ✅ First-class |
| Streaming support | ⚠️ Limited | ⚠️ Limited | ✅ Native |
| Best for | Small APIs | Full web apps | APIs + AI backends |

FastAPI is the right choice here because this project is an **API backend** — not a website — and we need streaming, validation, and speed.

### The Chat Endpoint

```python
@app.post("/chat")
def chat(request: ChatRequest):
    answer = ask_llm(request.message)
    return {"response": answer}
```

Line by line:

| Line | What It Does |
|---|---|
| `@app.post("/chat")` | Registers this function to handle POST requests at `/chat` |
| `def chat(request: ChatRequest):` | FastAPI sees `ChatRequest` type hint and automatically parses + validates the incoming JSON body |
| `answer = ask_llm(request.message)` | Passes the validated message to our business logic in `services.py` |
| `return {"response": answer}` | FastAPI automatically converts this Python dict to a JSON response |

### Swagger UI

FastAPI auto-generates interactive API documentation at `http://localhost:8000/docs`.

```
http://localhost:8000/docs   ← Try your endpoints in a browser
http://localhost:8000/redoc  ← Alternative documentation view
```

Why is Swagger useful?

- Test API endpoints without writing a single line of frontend code
- See exactly what request body shape is required
- See exactly what the response looks like
- Share the docs URL with anyone who needs to integrate your API

---

## 📐 Pydantic Schemas

```python
# schemas.py
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
```

### What Is This Doing?

`ChatRequest` defines the **expected shape** of every incoming request body.

When a POST request arrives at `/chat`, FastAPI reads the JSON body and tries to create a `ChatRequest` object. If the body doesn't have a `message` field, or if `message` isn't a string, FastAPI **automatically returns a 422 error** before your code ever runs.

### Why Schemas Matter

| Without Schema | With Schema |
|---|---|
| You manually check `if "message" in data` | FastAPI checks automatically |
| Wrong types crash your code silently | Wrong types return a clear error immediately |
| No auto-documentation | Swagger knows exactly what to show |
| You write validation logic yourself | Pydantic handles it for free |

---

## 🔀 Separation of Concerns — `main.py` / `services.py`

### Why Not Write Everything in `main.py`?

Bad practice — everything in routes:

```python
@app.post("/chat")
def chat(request: ChatRequest):
    # 50 lines of business logic here
    # hard to test, hard to reuse, hard to read
```

Good practice — separated concerns:

```python
# main.py — only handles HTTP
@app.post("/chat")
def chat(request: ChatRequest):
    answer = ask_llm(request.message)   # delegates to services.py
    return {"response": answer}

# services.py — only handles business logic
def ask_llm(message: str) -> str:
    response = ollama.chat(...)
    return response
```

### Flow

```
main.py          ← Handles HTTP only (routes, request parsing, response)
    │
    ▼
services.py      ← Handles business logic only (talks to Ollama)
    │
    ▼
Ollama           ← Handles model inference only
    │
    ▼
Return response
```

Each layer has **one responsibility**. If you want to swap Ollama for a different model, you only change `services.py` — `main.py` doesn't know or care.

---

## 🤖 Ollama Integration

### What Is Ollama?

Ollama is a tool that lets you **run large language models locally** on your own machine. Think of it as a local version of the ChatGPT backend — it serves AI models via a simple API running at `localhost:11434`.

### Why Local Inference?

| Cloud API (OpenAI, etc.) | Ollama (Local) |
|---|---|
| Costs money per token | Free after setup |
| Your data goes to their servers | Everything stays on your machine |
| Requires internet | Works offline |
| Rate limits apply | No limits |
| Best for production | Best for development & private use |

### The Ollama Call

```python
import ollama

response = ollama.chat(
    model="qwen3:4b",
    messages=chat_history,
    stream=False
)
```

| Parameter | What It Does |
|---|---|
| `model="qwen3:4b"` | Specifies which model to run. Qwen3:4B is 4 billion parameters — powerful enough for most tasks, small enough to run on a laptop |
| `messages=chat_history` | The full conversation history in a structured format (see below) |
| `stream=False` | Return the complete response at once (vs token by token) |

**Concepts to know:**

- **Temperature** — Controls randomness. Low (0.1) = deterministic, focused. High (1.0) = creative, varied. Most chatbots use 0.7.
- **Context Window** — How many tokens the model can "see" at once. Qwen3:4B has a limited context window, so very long conversations may lose early context.

---

## 💬 Chat Message Format

Ollama (and most LLMs) expect messages in this exact structure:

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "What is Python?"
    },
    {
        "role": "assistant",
        "content": "Python is a high-level programming language..."
    },
    {
        "role": "user",
        "content": "Give me an example."
    }
]
```

### The Three Roles

| Role | Who Sends It | Purpose |
|---|---|---|
| `system` | You (the developer) | Sets the model's persona, behavior, and rules. The user never sees this directly. |
| `user` | The human chatting | The human's actual message |
| `assistant` | The model | The model's previous responses — so it remembers what it said |

The model reads the **entire message list** before generating each response, which is what gives it context.

---

## 🔒 System Prompt

```python
{
    "role": "system",
    "content": "You are a helpful assistant."
}
```

### Why Does the System Prompt Exist?

The system prompt is how you **instruct the model** before the user ever types anything. Examples:

- `"You are a customer support agent for AcmeCorp. Only answer questions about our products."`
- `"You are a Python tutor. Explain everything simply. Always include code examples."`
- `"Answer only in French."`

### Why Can't the User Override It?

The system role has the **highest instruction priority** in the message hierarchy. While a user could *ask* the model to ignore it, a well-crafted system prompt resists this — and you control what gets injected into the `system` role on the backend. The user sends `user` role messages; they never touch the `system` slot.

---

## 🧠 Chat History

### Why Is History Needed?

Without history, every request is a fresh conversation:

```
User:      "My name is Jeevith."
Assistant: "Nice to meet you, Jeevith!"

[New request — model has forgotten everything]

User:      "What's my name?"
Assistant: "I don't know your name."   ← Wrong
```

With history, the full conversation travels with every request:

```python
chat_history = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "My name is Jeevith."},
    {"role": "assistant", "content": "Nice to meet you, Jeevith!"},
    {"role": "user",      "content": "What's my name?"},
]
# Model sees all 4 messages → answers correctly
```

### How It Works in Code

```python
# After user sends a message
chat_history.append({"role": "user", "content": user_message})

# After model responds
chat_history.append({"role": "assistant", "content": model_response})
```

Both the user message AND the assistant reply get stored — so the model always knows what was said by whom.

---

## 🌊 Streaming Responses

### Without Streaming

```
User sends message
        │
        │   [waiting 5–15 seconds...]
        │
Model finishes generating 500 words
        │
Full response appears all at once
```

The user stares at a blank screen. Bad experience.

### With Streaming

```
User sends message
        │
        ▼
"The"          ← token 1 arrives instantly
"Python"       ← token 2
"language"     ← token 3
"was"          ← token 4
...            ← continues until done
```

The user sees output immediately, just like ChatGPT's typing effect.

### How Streaming Works in Code

```python
# Enable streaming from Ollama
response = ollama.chat(
    model="qwen3:4b",
    messages=chat_history,
    stream=True   # ← Ollama now yields chunks, not a full response
)

# Generator function that yields each chunk
def generate():
    for chunk in response:
        token = chunk["message"]["content"]
        yield token

# FastAPI streams the generator output to the client
return StreamingResponse(generate(), media_type="text/plain")
```

---

## ⚙️ Generator Functions

### `return` vs `yield`

```python
# return — computes everything, sends it all at once
def get_all_tokens():
    tokens = []
    for chunk in ollama_stream:
        tokens.append(chunk)
    return tokens   # ← entire list built in memory first, then returned

# yield — sends each item as it arrives, never builds a full list
def stream_tokens():
    for chunk in ollama_stream:
        yield chunk   # ← each chunk sent immediately, memory stays flat
```

| `return` | `yield` |
|---|---|
| Builds full result in memory | Never stores more than one chunk |
| Caller gets everything at once | Caller gets items one at a time |
| Bad for streaming | Perfect for streaming |
| Bad for huge datasets | Memory efficient |

### Execution Flow with `yield`

```
stream_tokens() called
        │
        ▼
First chunk from Ollama arrives → yielded → sent to browser
        │
Second chunk arrives → yielded → sent to browser
        │
...continues until Ollama stream ends
        │
Function exits naturally
```

The function is **paused and resumed** at each `yield` — it never blocks.

### Why `StreamingResponse`?

`StreamingResponse` tells FastAPI: "don't wait for this function to finish — send each yielded chunk to the client as it arrives." Without it, FastAPI would wait for the generator to complete before sending anything — defeating the purpose.

### Sync vs Async

- **Synchronous** — one task runs, everything else waits
- **Asynchronous** — tasks can pause and let others run while waiting (e.g. waiting for Ollama)

Streaming works without `async` for a single user. When **multiple users** connect simultaneously, `async def` becomes important — it lets FastAPI handle other requests while one user's Ollama stream is still running, rather than blocking everyone.

### Final Streaming Architecture

```
Browser
  │  POST /chat
  ▼
FastAPI (main.py)
  │  StreamingResponse(generate())
  ▼
Generator Function (services.py)
  │  iterates over Ollama stream
  ▼
Ollama Stream
  │  runs Qwen3:4B model
  ▼
Model generates token-by-token
  │  each chunk yielded upward
  ▼
Chunk → Generator → StreamingResponse → Browser
(repeated until generation complete)
```

---

## 🔍 Phase 2 — RAG Pipeline

### What Problem Does RAG Solve?

The Qwen3:4B model only knows what it was trained on — data up to a certain date, no knowledge of your private files.

```
Without RAG:

User: "Summarize my company's refund policy."
LLM:  "I don't have access to your company's documents."
```

RAG (Retrieval-Augmented Generation) lets you **inject your own documents** into the model's context at query time:

```
With RAG:

User: "Summarize my company's refund policy."
  │
  ▼
Search ChromaDB for relevant document chunks
  │
  ▼
Retrieve: [chunk from refund_policy.pdf]
  │
  ▼
Inject into prompt: "Using this context: [chunk]... answer: Summarize..."
  │
  ▼
LLM: "Your refund policy states: full refund within 30 days..."  ✅
```

### What Are Hallucinations?

Without RAG, if you ask the model something it doesn't know, it may **confidently make up an answer** — this is called a hallucination. RAG grounds the model's answer in real retrieved text, dramatically reducing this.

---

## ⚖️ Fine-Tuning vs RAG

| | Fine-Tuning | RAG |
|---|---|---|
| **What it does** | Permanently changes model weights | Retrieves documents at query time |
| **Cost** | Very expensive (GPU hours) | Cheap to set up |
| **Hardware** | Requires powerful GPU | Runs on a laptop |
| **Updating knowledge** | Re-train from scratch | Just add new documents to ChromaDB |
| **Best for** | Changing model *behavior* or *style* | Giving model access to *specific knowledge* |
| **Risk** | Can break existing capabilities | No model changes at all |
| **Company documents** | ❌ Slow, expensive, risky | ✅ Ideal |

**In short:** Fine-tuning teaches the model a new personality. RAG teaches the model to look things up.

### Why We Add RAG to This Project Instead of Starting Fresh

The chatbot already has FastAPI, Ollama, streaming, and history working perfectly. RAG is an **additional layer** slotted in before the Ollama call:

```
Current flow:         User → FastAPI → Ollama → Answer

RAG flow:             User → FastAPI → ChromaDB search
                                            │
                                     Relevant chunks
                                            │
                                          Ollama → Grounded answer
```

The existing infrastructure stays completely intact. RAG is additive.

---

## 🏗 RAG Architecture

```
User Question
      │
      ▼
FastAPI (/chat endpoint)
      │
      ▼
Embed the question (Sentence Transformers)
      │
      ▼
Search ChromaDB for similar chunks
      │
      ▼
Retrieve top-N relevant document chunks
      │
      ▼
Build enriched prompt:
  "Context: [retrieved chunks]
   Question: [user question]"
      │
      ▼
Ollama + Qwen3:4B
      │
      ▼
Answer grounded in your documents
```

---

## 📄 PDF Reader

The RAG pipeline starts with getting text out of your documents.

```python
# rag.py
from pypdf import PdfReader

def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)  # Open and parse the PDF file
    text = ""

    for page in reader.pages:     # Iterate over every page
        page_text = page.extract_text()

        if page_text:              # Some pages are images — skip if no text extracted
            text += page_text + "\n"  # Append page text, add newline between pages

    return text                    # Return full document as one string
```

Line-by-line breakdown:

| Line | Explanation |
|---|---|
| `PdfReader(file_path)` | Opens the PDF and parses its internal structure (PDFs are not plain text — they have a complex binary format) |
| `reader.pages` | A list of page objects, one per page in the document |
| `for page in reader.pages` | Loop through every page |
| `page.extract_text()` | Pulls the text content from the page. May return `None` if the page is a scanned image with no embedded text |
| `if page_text:` | Guards against `None` — some PDF pages contain only images, charts, or blank space |
| `text += page_text + "\n"` | Accumulates all text into a single string, with newlines separating pages |
| `return text` | Returns the complete document text for further processing |

### How It Was Tested

```python
# test.py
from app.rag import read_pdf

text = read_pdf("documents/resume.pdf")
print(text)
```

Running this and seeing the resume content printed confirms that:
- The PDF is being opened correctly
- Text is being extracted from every page
- The string is being assembled properly
- The function is ready to feed into the chunking step

---

## ✅ Current Progress

| Feature | Status |
|---|---|
| FastAPI REST API | ✅ Complete |
| Ollama Integration | ✅ Complete |
| Qwen3:4B Local Model | ✅ Complete |
| Chat History | ✅ Complete |
| Streaming Responses | ✅ Complete |
| Generator Functions | ✅ Complete |
| PDF Text Extraction | ✅ Complete |
| Text Chunking | 🔄 Next |
| Embeddings (Sentence Transformers) | 📋 Planned |
| ChromaDB Vector Storage | 📋 Planned |
| Similarity Search | 📋 Planned |
| Context Injection into Prompt | 📋 Planned |
| Complete RAG Pipeline | 📋 Planned |

---

## 🗺 Roadmap

### ✅ Done — Phase 1: Basic Chatbot
FastAPI · Ollama · Streaming · Chat History

### ✅ Done — Phase 2 (Partial): RAG Foundation
PDF reading with `pypdf`

### 🔄 Next — Chunking

Before embedding documents, long text must be split into smaller chunks.
A 30-page PDF can't fit in the model's context window — so we split it into overlapping segments using `RecursiveCharacterTextSplitter`.

```python
# Coming next
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(raw_text)
```


## 🔗 Phase 3 — RAG Integration (Complete Pipeline)

> The chatbot is now a full **Retrieval-Augmented Generation system** — every answer is grounded in your indexed documents, not just the model's training data.

### What Changed

| Before RAG Integration | After RAG Integration |
|---|---|
| User question → Ollama → Answer | User question → ChromaDB search → Relevant chunks → Ollama → Grounded answer |
| Model answers from training data only | Model answers from **your documents** |
| Risk of hallucination is high | Hallucination dramatically reduced |
| Generic responses | Specific, document-backed responses |

---

### RAG Request Flow

```
User Question
      │
      ▼
Generate Query Embedding
(Sentence Transformers converts question to a vector)
      │
      ▼
ChromaDB Semantic Search
(find chunks whose vectors are closest to the question vector)
      │
      ▼
Retrieve Top-K Relevant Chunks
(the most semantically similar document segments)
      │
      ▼
Build Context String
(join all chunks into one block of text)
      │
      ▼
Construct Enriched Prompt
(context + question wrapped in instructions)
      │
      ▼
Ollama — Qwen3:4B
(model reads context before answering)
      │
      ▼
Stream Generated Answer → Client
```

---

### Step 1 — Semantic Search

```python
chunks = semantic_search(question)
```

The user's question is embedded into a vector, then ChromaDB finds the stored chunks whose vectors are closest in meaning — not just matching keywords, but actual semantic similarity.

---

### Step 2 — Building Context

```python
context = "\n\n".join(chunks)
```

All retrieved chunks are merged into a single string. This becomes the **knowledge source** the model will read before generating its answer.

Why join with `"\n\n"`? Double newlines clearly separate each chunk visually inside the prompt, making it easier for the model to distinguish where one chunk ends and another begins.

---

### Step 3 — Prompt Engineering

Instead of sending only the user's question to Ollama, a **structured prompt** wraps the context around it:

```
You are a helpful AI assistant.

Answer ONLY using the provided context.
If the answer is not available in the context, respond with "I don't know."

Context:
{retrieved chunks injected here}

Question:
{user question}

Answer:
```

Why this structure matters:

| Instruction | Why It's There |
|---|---|
| `"Answer ONLY using the provided context"` | Forces the model to use retrieved text, not hallucinate from training data |
| `"If not available, say I don't know"` | Prevents confident wrong answers — honesty over hallucination |
| `Context:` block | Gives the model factual ground to stand on |
| `Question:` block | Clearly separates the actual query from the context |
| `Answer:` label | Signals to the model exactly where its response should begin |

---

### Step 4 — Updated Chat Service Flow

```
services.py now runs this sequence:

1. Receive user question
        │
        ▼
2. semantic_search(question)      ← query ChromaDB
        │
        ▼
3. context = "\n\n".join(chunks)  ← build context string
        │
        ▼
4. Construct enriched prompt      ← wrap context + question
        │
        ▼
5. ollama.chat(prompt, stream=True)
        │
        ▼
6. yield chunks → StreamingResponse → Browser
```

The model no longer answers from pretrained knowledge alone — it reads the retrieved context first, every single time.

---

### Worked Example

**User asks:**
```
What programming languages does Jeevith know?
```

**ChromaDB retrieves this chunk from the indexed resume:**
```
Programming Languages:
Java (Advanced)
Python (Expert)
C (Intermediate)
```

**Prompt sent to Ollama:**
```
You are a helpful AI assistant.
Answer ONLY using the provided context.
If the answer is not available in the context, respond with "I don't know."

Context:
Programming Languages:
Java (Advanced)
Python (Expert)
C (Intermediate)

Question:
What programming languages does Jeevith know?

Answer:
```

**Model response:**
```
Jeevith is proficient in Java, Python, and C.
```

The model answered **only from the document** — no guessing, no hallucination.

---

### Why RAG Beats a Plain LLM

```
Without RAG:                      With RAG:

User Question                     User Question
      │                                 │
      ▼                                 ▼
     LLM                         Semantic Search
      │                                 │
      ▼                                 ▼
Generic / hallucinated          Relevant Context
answer from training data               │
                                        ▼
                                       LLM
                                        │
                                        ▼
                                Accurate, document-backed answer
```

---

## 🏗 Final Project Architecture

```
app/
│
├── main.py          ← FastAPI routes — HTTP in, HTTP out
├── services.py      ← RAG-aware chat service — orchestrates the full pipeline
├── rag.py           ← PDF reading + recursive text chunking
├── embeddings.py    ← Sentence Transformer — text → vectors
├── vector_db.py     ← ChromaDB — store, index, and search vectors
│
├── documents/       ← Drop your PDFs here
└── chroma_db/       ← ChromaDB persists vector data here
```

---

## ✅ Complete Capability Summary

| Capability | Status |
|---|---|
| FastAPI REST API | ✅ Complete |
| Ollama local inference | ✅ Complete |
| Qwen3:4B model | ✅ Complete |
| Chat history | ✅ Complete |
| Streaming responses | ✅ Complete |
| PDF document ingestion | ✅ Complete |
| Recursive text chunking | ✅ Complete |
| Sentence Transformer embeddings | ✅ Complete |
| ChromaDB vector storage | ✅ Complete |
| Semantic similarity search | ✅ Complete |
| Context-aware prompt construction | ✅ Complete |
| Full RAG pipeline | ✅ Complete |

---

## 🗺 What's Next

The core RAG pipeline is complete. Future extensions could include:

- **Multi-document support** — index an entire folder of PDFs at once
- **Re-ranking** — score retrieved chunks by relevance before injecting
- **Hybrid search** — combine semantic search with keyword search for better recall
- **Chat history + RAG** — persist both conversation memory and document context simultaneously
- **Web UI** — a frontend to chat with documents visually
- **Authentication** — secure the API for multi-user environments


---

## 📬 Contact

<div align="center">

[![Portfolio](https://img.shields.io/badge/🌐%20Portfolio-0a0e1a?style=for-the-badge)](https://django-portfolio-yd0b.onrender.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0a0e1a?style=for-the-badge&logo=linkedin&logoColor=64FFDA)](https://www.linkedin.com/in/jeevith-swarup-tuta-284607345/)
[![Gmail](https://img.shields.io/badge/Gmail-0a0e1a?style=for-the-badge&logo=gmail&logoColor=64FFDA)](mailto:jeevithswaruptuta@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-0a0e1a?style=for-the-badge&logo=github&logoColor=64FFDA)](https://github.com/jeevithswarup)

<br/>

**Jeevith Swarup** — Backend Developer | Building local AI infrastructure from scratch.

</div>

---
