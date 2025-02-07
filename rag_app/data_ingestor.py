import os
from typing import List
from uuid import uuid4

from qdrant_client import QdrantClient, models
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_app.config import Config
from rag_app.file_loader import File

# TODO: cambiar para proporcionar varios documentos como contexto??
CONTEXT_PROMPT = ChatPromptTemplate.from_template(
    """
    Eres un experto en pliegos. Tu tarea es que respondas a las preguntas que se te hagan utilizando el contexto relevante de cada chunk
    
    Este es el documento:
    <document>
    {document}
    </document>
    
    Este es el chunk:
    <chunk>
    {chunk}
    </chunk>
    """.strip()
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=Config.Preprocessing.CHUNK_SIZE,
    chunk_overlap=Config.Preprocessing.CHUNK_OVERLAP
)

def create_llm() -> ChatOllama:
    return ChatOllama(model=Config.Preprocessing.LLM, temperature=0, keep_alive=1)

def create_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(model_name=Config.Preprocessing.EMBEDDING_MODEL)

def create_reranker() -> FlashrankRerank:
    return FlashrankRerank(model=Config.Preprocessing.RERANKER, top_n=Config.Chatbot.N_CONTEXT_RESULTS)

def _generate_context(llm: ChatOllama, document: str, chunk: str) -> str:
    messages = CONTEXT_PROMPT.format_messages(document=document, chunk=chunk)
    response = llm.invoke(messages)
    return response.content

def _create_chunks(document: str) -> List[Document]:
    chunks = text_splitter.split_documents([document])
    if not Config.Preprocessing.CONTEXTUALIZE_CHUNKS:
        return chunks
    llm = create_llm()
    contextual_chunks = []
    for chunk in chunks:
        context = _generate_context(llm, document.page_content, chunk.page_content)
        chunk_with_context = f"{context}\n\n{chunk.page_content}"
        contextual_chunks.append(Document(page_content=chunk_with_context, metadata=chunk.metadata))
    return contextual_chunks

# TODO: modificar la funcion para que si no recibe archivos, utilice los del index y no agregue ningun chunk
def ingest_files(files: List[File]) -> BaseRetriever:
    if files:
        documents = [Document(file.content, metadata={"source": file.name}) for file in files]
        chunks = []
        for document in documents:
            chunks.extend(_create_chunks(document))
        uuids = [str(uuid4()) for _ in range(len(documents))]

        print("Adding chunks to Qdrant collection...")
        vector_store.add_documents(documents=chunks, ids=uuids)
        print("Chunks added to Qdrant collection")

    """
    if files:
        documents = [Document(file.content, metadata={"source": file.name}) for file in files]
        chunks = []
        for document in documents:
            chunks.extend(_create_chunks(document))

        print("Adding chunks to FAISS index...")
        faiss_index.add_documents(chunks)
        print("Chunks added to FAISS index")

    semantic_retriever = faiss_index.as_retriever(
        search_kwargs={"k": Config.Preprocessing.N_SEMANTIC_RESULTS}
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = Config.Preprocessing.N_BM25_RESULTS

    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4]
    )

    return ContextualCompressionRetriever(
        base_compressor=create_reranker(), base_retriever=ensemble_retriever
    )
    """

print("Authenticating into qdrant")
qdrant_client = QdrantClient(
    "https://dccbf4cf-85a8-48a9-9a9b-36edad0751d1.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ2NjkyMDE5fQ.lcSfk1W2EOThyflEg1q6WJMWck7eAvlV5q7MrD-YVfM")
print("User authenticated to qdrant cluster")

# Create the collection if it does not exist
collection = qdrant_client.collection_exists(Config.Database.COLLECTION_NAME)
if not collection:
    qdrant_client.create_collection(
        collection_name=Config.Database.COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )

vector_store = QdrantVectorStore(
    client=qdrant_client,
    embedding=create_embeddings(),
    collection_name=Config.Database.COLLECTION_NAME,
)

def create_retriever() -> BaseRetriever:
    print("Creating retriever without new files")
    return vector_store.as_retriever(
        search_kwargs={"k": Config.Preprocessing.N_SEMANTIC_RESULTS}
    )
