import uuid
from app.database.database import conn


def create_session():
    session_id = str(uuid.uuid4())

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO sessions (session_id)
        VALUES (%s)
        """,
        (session_id,)
    )

    cursor.close()

    return session_id