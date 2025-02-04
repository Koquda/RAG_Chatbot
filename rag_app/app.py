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
st.subheader("Inteligencia privada sobre ASAC")

@st.cache_resource(show_spinner=False)
def create_chatbot(files: List[UploadedFile]):
    files = [load_uploaded_file(file) for file in files]
    return Chatbot(files)

def show_upload_documents() -> List[UploadedFile]:
    holder = st.empty()
    with holder.container():
        uploaded_files = st.file_uploader(
            label="Sube tus documentos", type=["pdf", "txt", "md"], accept_multiple_files=True
        )
    if not uploaded_files:
        st.warning("Por favor, sube tus documentos")
        st.stop()

    with st.spinner("Analizando documentos..."):
        holder.empty()
        return uploaded_files


uploaded_files = show_upload_documents()
chatbot = create_chatbot(uploaded_files)

if "messages" not in st.session_state:
    st.session_state.messages = create_history(WELCOME_MESSAGE)

with st.sidebar:
    st.title("Tus documentos")
    file_list_text = "\n".join(f"- {file.name}" for file in chatbot.files)
    st.markdown(file_list_text)

for message in st.session_state.messages:
    avatar = "🐧" if message.role == Role.USER else "🤖"
    with st.chat_message(message.role.value, avatar=avatar):
        st.markdown(message.content)

if prompt := st.chat_input("Escribe tu pregunta..."):
    with st.chat_message("user", avatar="🐧"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        full_response = ""
        message_placeholder = st.empty()
        message_placeholder.status(random.choice(LOADING_MESSAGES), state="running")
        for event in chatbot.ask(prompt, st.session_state.messages):
            if isinstance(event, SourcesEvent):
                for i, doc in enumerate(event.content):
                    with st.expander(f"Fuente {i + 1}"):
                        st.markdown(doc.page_content)
            if isinstance(event, ChunkEvent):
                chunk = event.content
                full_response += chunk
                message_placeholder.markdown(full_response)