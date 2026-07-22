from app.database.database import conn


def get_messages(session_id: str):
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, content
        FROM messages
        WHERE session_id = %s
        ORDER BY created_at;
        """,
        (session_id,)
    )

    rows = cursor.fetchall()
    cursor.close()

    messages = []

    for role, content in rows:
        messages.append(
            {
                "role": role,
                "content": content
            }
        )

    return messages


def save_message(session_id, role, content):
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO messages (session_id, role, content)
        VALUES (%s, %s, %s)
        """,
        (session_id, role, content)
    )

    conn.commit()
    cursor.close()