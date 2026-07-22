from app.database.session_manager import create_session
from app.database.message_manager import save_message, get_messages

session = create_session()

save_message(session, "user", "Hello")
save_message(session, "assistant", "Hi! How can I help you?")

print(get_messages(session))