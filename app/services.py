import ollama
from app.vector_db import semantic_search
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

conversations = {}
chat_history = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]

conversations = {}

def stream_llm( conversation_id: str,question: str, context: str):
    messages = chat_history.copy()

    messages.append({
        "role": "user",
        "content": f"""
Context:
{context}

Question:
{question}

Answer:
"""
    })

    stream = ollama.chat(
        model="qwen3:4b",
        messages=messages,
        stream=True
    )

    answer = ""

    for chunk in stream:
        text = chunk["message"]["content"]
        answer += text
        yield text

    # Store only the conversation, not the retrieved context
    chat_history.append({
        "role": "user",
        "content": question
    })

    chat_history.append({
        "role": "assistant",
        "content": answer
    })


def chat(conversation_id: str, question: str):

    if conversation_id not in conversations:
        conversations[conversation_id] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]

    chunks = semantic_search(question)

    context = "\n\n".join(chunks)

    return stream_llm(
        conversation_id,
        question,
        context
    )