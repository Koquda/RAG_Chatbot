import random
from typing import List

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from rag_app.chatbot import Chatbot, ChunkEvent, Message, Role, SourcesEvent, create_history
from rag_app.file_loader import load_uploaded_file

LOADING_MESSAGES = [
    "Cargando...",
    "Estoy pensando...",
    "Un momento...",
    "Estoy buscando...",
    "Dame un segundo...",
    "Estoy procesando...",
    "Estoy trabajando en ello...",
    "Estoy calculando...",
    "Estoy generando...",
]

WELCOME_MESSAGE = Message(role=Role.ASSISTANT, content="¡Hola! Soy tu asistente virtual. ¿En qué puedo ayudarte hoy?")

st.set_page_config(
    page_title="ASAC Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.header("ASAC Chatbot")

@st.cache_resource(show_spinner=False)
def create_chatbot(files: List[UploadedFile]):
    # Load the files before creating the Chatbot
    files = [load_uploaded_file(file) for file in files]
    return Chatbot(files)

# --- File uploader logic ---
# We'll use session state to control whether the file uploader is shown.
if "show_file_uploader" not in st.session_state:
    st.session_state.show_file_uploader = False
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = None

def process_uploaded_files():
    """
    Process the uploaded files and create the chatbot.
    """
    if st.session_state.uploaded_files is None:
        st.warning("Por favor, sube tus documentos antes de continuar.")
        st.stop()
    
    with st.spinner("Analizando documentos..."):
        return create_chatbot(st.session_state.uploaded_files)

# --- Chat input and file uploader button side by side ---
# Create two columns: one for chat input, one for the upload button.
cols = st.columns([4, 1])

with cols[0]:
    # You can use st.chat_input (if you are using Streamlit version that supports it)
    # or a standard text_input. Here we use st.chat_input:
    prompt = st.chat_input("Escribe tu pregunta...")

with cols[1]:
    # When the button is pressed, reveal the file uploader widget.
    if st.button("Subir archivos"):
        st.session_state.show_file_uploader = True

# If the uploader should be shown, display it right below the chat input.
if st.session_state.show_file_uploader:
    st.markdown("### Sube tus documentos")
    # The file uploader widget; note that it accepts multiple files.
    uploaded_files = st.file_uploader(
        label="Selecciona tus archivos",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="uploader"
    )
    if uploaded_files:
        if st.session_state.uploaded_files is None:
            st.session_state.uploaded_files = []
        st.session_state.uploaded_files.extend(uploaded_files)
        # Optionally, once files are uploaded, you might want to hide the uploader again:
        st.session_state.show_file_uploader = False

# Before processing chat messages, ensure that files have been uploaded.
if st.session_state.uploaded_files is None:
    st.info("Por favor, sube tus documentos usando el botón de la derecha para comenzar.")
    st.stop()

# Create the chatbot once files have been uploaded.
chatbot = process_uploaded_files()

# --- Chat history and sidebar ---
if "messages" not in st.session_state:
    st.session_state.messages = create_history(WELCOME_MESSAGE)

with st.sidebar:
    st.title("Tus documentos")
    # Display the names of the uploaded documents.
    file_list_text = "\n".join(f"- {file.name}" for file in chatbot.files)
    st.markdown(file_list_text)

for message in st.session_state.messages:
    avatar = "🐧" if message.role == Role.USER else "🤖"
    with st.chat_message(message.role.value, avatar=avatar):
        st.markdown(message.content)

# --- Process the chat prompt ---
if prompt:
    # Add user message to chat history
    with st.chat_message("user", avatar="🐧"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        full_response = ""
        message_placeholder = st.empty()
        # Display a random loading message
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")
        # Process the chatbot response chunk by chunk
        for event in chatbot.ask(prompt, st.session_state.messages):
            if isinstance(event, SourcesEvent):
                for i, doc in enumerate(event.content):
                    with st.expander(f"Fuente {i + 1}"):
                        st.markdown(doc.page_content)
            if isinstance(event, ChunkEvent):
                chunk = event.content
                full_response += chunk
                message_placeholder.markdown(full_response)
