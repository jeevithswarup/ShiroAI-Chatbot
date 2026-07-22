import ollama

from app.vector_db import semantic_search
from app.database.message_manager import (
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


def build_search_query(session_id: str, question: str):
    messages = get_messages(session_id)

    history = []

    for msg in messages[-4:]:
        history.append(f"{msg['role']}: {msg['content']}")

    history.append(f"user: {question}")

    return "\n".join(history)


def stream_llm(session_id: str, question: str, context: str):
    # Load previous conversation
    messages = get_messages(session_id)

    # Save current user message to the database
    save_message(session_id, "user", question)

    # Messages sent to the LLM
    llm_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Add previous conversation history
    llm_messages.extend(messages)

    # Add the current user message
    llm_messages.append(
        {
            "role": "user",
            "content": f"""
You have two sources of information:

1. Previous conversation history.
2. Resume context.

Rules:
- If the user refers to previous messages (for example: "What was my first question?", "What did I ask before?", "Summarize our conversation"), use the conversation history.
- If the user asks about the resume, use ONLY the resume context.
- If the answer is not in either the conversation history or the resume context, reply with "I don't know."

Resume Context:
{context}

Current Question:
{question}
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
    save_message(session_id, "assistant", answer)
    
def chat(session_id: str, question: str):
    search_query = build_search_query(
        session_id,
        question,
    )

    chunks = semantic_search(search_query)

    context = "\n\n".join(chunks)

    return stream_llm(
        session_id,
        question,
        context,
    )