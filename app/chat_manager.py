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


def get_messages(conversation_id: str):
    if conversation_id not in conversations:
        conversations[conversation_id] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            }
        ]
    return conversations[conversation_id]

messages = get_messages(session_id)

if not messages:
    llm_messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]
else:
    llm_messages = messages