import ollama

from app.vector_db import semantic_search
from app.chat_manager import (
    get_messages,
    save_message,
)

SYSTEM_PROMPT = """
You are a Resume Analysis AI.

Use ONLY the provided context to answer questions.

If the user asks for:
- ATS score
- Resume feedback
- Job eligibility
- Missing skills
- Resume improvements

you may analyze the resume and provide your own professional assessment.

If the answer cannot be determined from the resume or your analysis, say you don't know.

Do not make up facts that are not supported by the resume.
"""


def build_search_query(conversation_id: str, question: str):
    messages = get_messages(conversation_id)

    history = []

    for msg in messages[-4:]:
        if msg["role"] != "system":
            history.append(f'{msg["role"]}: {msg["content"]}')

    history.append(f"user: {question}")

    return "\n".join(history)


def stream_llm(conversation_id: str, question: str, context: str):
    # Load previous conversation
    messages = get_messages(conversation_id)

    # Save current user message to database
    save_message(conversation_id, "user", question)

    # Messages sent to the LLM
    llm_messages = messages.copy()

    llm_messages.append(
        {
            "role": "user",
            "content": f"""
Context:
{context}

Question:
{question}

Answer:
"""
        }
    )

    stream = ollama.chat(
        model="qwen3:4b",
        messages=llm_messages,
        stream=True,
    )

    answer = ""

    for chunk in stream:
        text = chunk["message"]["content"]
        answer += text
        yield text

    # Save assistant reply
    save_message(conversation_id, "assistant", answer)


def chat(conversation_id: str, question: str):
    search_query = build_search_query(
        conversation_id,
        question,
    )

    chunks = semantic_search(search_query)

    context = "\n\n".join(chunks)

    return stream_llm(
        conversation_id,
        question,
        context,
    )