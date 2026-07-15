import psycopg2

conn = psycopg2.connect(
    host="localhost",
    port="5432",
    database="Ai-chatbot",
    user="postgres",
    password="Test@123"
)

cursor = conn.cursor()