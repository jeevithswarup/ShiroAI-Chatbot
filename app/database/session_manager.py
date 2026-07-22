
import uuid
from .database import conn
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

    conn.commit()
    cursor.close()

    return session_id