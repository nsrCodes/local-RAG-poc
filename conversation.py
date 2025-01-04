import argparse
import shutil
import os
import sqlite3
import hashlib
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM as Ollama
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.memory import ConversationBufferMemory


# UTILS #
def compute_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def is_cached(file_hash):
    conn = sqlite3.connect("pdf_index.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS pdfs (hash TEXT PRIMARY KEY, path TEXT)")
    c.execute("SELECT 1 FROM pdfs WHERE hash = ?", (file_hash,))
    res = c.fetchone()
    conn.close()
    return bool(res)

def store_hash(file_hash, pdf_path):
    conn = sqlite3.connect("pdf_index.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS pdfs (hash TEXT PRIMARY KEY, path TEXT)")
    c.execute("INSERT OR IGNORE INTO pdfs (hash, path) VALUES (?, ?)", (file_hash, pdf_path))
    conn.commit()
    conn.close()

def remove_hash(file_hash):
    conn = sqlite3.connect("pdf_index.db")
    c = conn.cursor()
    c.execute("DELETE FROM pdfs WHERE hash = ?", (file_hash,))
    conn.commit()
    conn.close()

def get_file_name(file_hash): 
    return f'temp_{file_hash}'

# RAG #
def get_embedding_function():
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings

def create_embeddings(db, documents, file_hash):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(documents)

    # Create a temporary Chroma DB.
    temp_path = get_file_name(file_hash)
    db.add_documents(chunks)
    # db.persist()

# Define prompt template
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an assistant that retrieves relevant information for the user's query. "
        "Use both the conversation history and the additional retrieved context to provide accurate and helpful answers. "
        "Follow the reasoning and structure instructions carefully to ensure your response is insightful and well-structured."
        "Remember to keep your responses concise unless specifically asked for additional details."
    ),
    HumanMessagePromptTemplate.from_template(
        "Conversation history:\n{chat_history}\n\n"
        "Additional retrieved context:\n{retrieved_context}\n\n"
        "User's current question:\n{query}\n\n"
        "---\n\n"
    ),
])

def query_rag(query_text: str, db, memory):
    results = db.similarity_search_with_score(query_text, k=10)

    # Step 2: Retrieve context from the database
    retrieved_context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Step 3: Build the prompt

    ## CONVERSATIONAL TEMPLATE ##
    prompt = PROMPT_TEMPLATE.format(
        chat_history="\n".join(
            [f"User: {msg.content}" if msg.type == "human" else f"Assistant: {msg.content}" for msg in memory.chat_memory.messages]
        ),
        retrieved_context=retrieved_context,
        query=query_text
    )

    # ## STATIC but better TEMPLATE ##
    # prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    # prompt = prompt_template.format(context=retrieved_context, question=query_text)

    # Step 4: Query the model
    model = Ollama(model="qwen2.5:7b-instruct-q8_0")
    response_text = ""
    for chunk in model.stream(prompt):
        print(chunk, end="", flush=True)
        response_text += chunk
    print("\n\n")
    # Step 5: Update memory with the new interaction
    memory.chat_memory.add_user_message(query_text)
    memory.chat_memory.add_ai_message(response_text)

    # Step 6: Format sources and response
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_sources = "\n".join(f"- {source}" for source in sources if source)
    formatted_response = f"\n\nResponse: {response_text}\n\nSources:\n{formatted_sources}"

    return formatted_response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file.")
    args = parser.parse_args()

    file_hash = compute_file_hash(args.pdf_path)
    db = Chroma(persist_directory=get_file_name(file_hash), embedding_function=get_embedding_function())

    if not is_cached(file_hash):
        print("📄 Loading PDF and creating embeddings...")
        loader = PyPDFLoader(args.pdf_path)
        documents = loader.load()

        create_embeddings(db, documents, file_hash)
        store_hash(file_hash, args.pdf_path)

    conversationMemory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    while True:
        user_question = input("Enter your question (or type 'exit' to quit): ")
        if user_question.lower() == 'exit':
            break
        print("\n")

        query_rag(user_question, db, conversationMemory)


    # Clean up.
    # shutil.rmtree(temp_path, ignore_errors=True)
    # remove_hash(file_hash)

if __name__ == "__main__":
    main()
