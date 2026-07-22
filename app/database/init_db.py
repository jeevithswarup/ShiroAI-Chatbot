from app.database.database import conn

def create_tables():
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
            role VARCHAR(20),
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cur.close()