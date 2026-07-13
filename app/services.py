import ollama
from app.vector_db import semantic_search

chat_history = [
    {
        "role": "system",
        "content": """
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
    }
]

def stream_llm(question: str, context: str):

    prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

    chat_history.append({
        "role": "user",
        "content": prompt
    })

    stream = ollama.chat(
        model="qwen3:4b",
        messages=chat_history,
        stream=True
    )

    answer = ""

    for chunk in stream:
        text = chunk["message"]["content"]
        answer += text
        yield text

    chat_history.append({
        "role": "assistant",
        "content": answer
    })


def chat(question: str):
    chunks = semantic_search(question)

    context = "\n\n".join(chunks)

    return stream_llm(question, context)