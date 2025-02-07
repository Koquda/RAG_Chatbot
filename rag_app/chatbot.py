from dataclasses import dataclass
from enum import Enum
from langchain.prompts import ChatPromptTemplate
from typing import Iterable, List, TypedDict
from rag_app.config import Config
from rag_app.data_ingestor import ingest_files, create_retriever
from rag_app.file_loader import File
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

SYSTEM_PROMPT = """
Estás teniendo una conversación con un usuario sobre fragmentos de sus archivos. Intenta ser útil y responder a sus preguntas.
Si no sabes la respuesta, di que no lo sabes e intenta hacer preguntas aclaratorias.
""".strip()

PROMPT = """
Esta es la información que tienes sobre los fragmentos de los documentos:

<context>
{context}
</context>

Un documento puede tener varios fragmentos.

Por favor, responde a la siguiente pregunta:

<question>
{question}
</question>

Respuesta:
"""

FILE_TEMPLATE = """
<file>
    <name>{name}</name>
    <content>{content}</content>
</file>
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PROMPT
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", PROMPT)
    ]
)

class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"

@dataclass
class Message:
    content: str
    role: Role

@dataclass
class ChunkEvent:
    content: str

@dataclass
class SourcesEvent:
    content: List[Document]

@dataclass
class FinalAnswerEvent:
    content: str

class State(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    context: List[Document]
    answer: str

def _remove_thinking_from_message(message: str) -> str:
    close_tag = "</think>"
    tag_length = len(close_tag)
    return message[message.find(close_tag) + tag_length :].strip()

def create_history(welcome_message: str) -> List[Message]:
    return [welcome_message]


class Chatbot:
    def __init__(self, files: List[File]):
        print("Creating chatbot")
        self.files = files
        self.retriever = create_retriever()
        self.ingest = ingest_files(files)
        self.llm = ChatOllama(
            model=Config.Model.NAME,
            temperature=Config.Model.TEMPERATURE,
            verbose=False,
            keep_alive=1
        )
        self.workflow = self._create_workflow()

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(
            FILE_TEMPLATE.format(name=doc.metadata["source"], content=doc.page_content)
            for doc in docs
        )

    def _retrieve(self, state: State):
        context = self.retriever.invoke(state["question"])
        return {"context": context}

    def _generate(self, state: State):
        messages = PROMPT_TEMPLATE.invoke(
            {
                "question": state["question"],
                "context": self._format_docs(state["context"]),
                "chat_history": state["chat_history"]
            }
        )
        answer = self.llm.invoke(messages)
        return {"answer": answer}
    
    def _create_workflow(self) -> CompiledStateGraph:
        graph_builder = StateGraph(State).add_sequence([self._retrieve, self._generate])
        graph_builder.add_edge(START, "_retrieve")
        return graph_builder.compile()
    
    def _ask_model(
        self, prompt: str, chat_history: List[Message]
    ) -> Iterable[SourcesEvent | ChunkEvent | FinalAnswerEvent]:
        history = [
            AIMessage(m.content) if m.role == Role.ASSISTANT else HumanMessage(m.content)
            for m in chat_history
        ]
        payload = {"question": prompt, "chat_history": history}

        config = {
            "configurable": {"thread_id": 42}
        }
        for event_type, event_data in self.workflow.stream(
            payload,
            config=config,
            stream_mode=["updates", "messages"]
        ):
            if event_type == "messages":
                chunk, _ = event_data
                yield ChunkEvent(chunk.content)
            if event_type == "updates":
                if "_retrieve" in event_data:
                    documents = event_data["_retrieve"]["context"]
                    yield SourcesEvent(documents)
                if "_generate" in event_data:
                    answer = event_data["_generate"]["answer"]
                    yield FinalAnswerEvent(answer.content)

    def ask(
        self, prompt: str, chat_history: List[Message]
    ) -> Iterable[SourcesEvent | ChunkEvent | FinalAnswerEvent]:
        for event in self._ask_model(prompt, chat_history):
            yield event
            if isinstance(event, FinalAnswerEvent):
                response = _remove_thinking_from_message("".join(event.content))
                chat_history.append(Message(role=Role.USER, content=prompt))
                chat_history.append(Message(role=Role.ASSISTANT, content=response))
