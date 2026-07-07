import ollama

chat_history = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant."
    }
]

def stream_llm(message: str):

    chat_history.append({
        "role": "user",
        "content": message
    })

    stream = ollama.chat(
        model="qwen3:4b",
        messages=chat_history,
        stream=True
    )

    answer = ""

    for chunk in stream:
        text = chunk.message.content
        answer += text
        yield text

    chat_history.append({
        "role": "assistant",
        "content": answer
    })