import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
import os
import dotenv

dotenv.load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
model = "meta-llama/llama-4-scout-17b-16e-instruct"
embedding_model = "mxbai-embed-large:latest"

# Does the ingestion, chunking, vectorize and storage step and retains store in session using streamlit
def pre_process_docs():
    if "retriever" not in st.session_state:
        st.session_state.doc_loader = PyPDFLoader(file_path="data/magic-tricks.pdf")
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        st.session_state.splitted_docs = st.session_state.doc_loader.load_and_split(text_splitter=st.session_state.splitter)
        st.session_state.embedding = OllamaEmbeddings(model=embedding_model)
        st.session_state.db = FAISS.from_documents(st.session_state.splitted_docs, embedding=st.session_state.embedding)
        st.session_state.llm = ChatGroq(api_key=groq_api_key, model=model)
        st.session_state.retriever = st.session_state.db.as_retriever(k=2)

def query_db_and_llm(prompt: str) -> str:
    template = PromptTemplate.from_template('''
    You are a joyous, happy and helpful magician assistant who only answers to questions from the provided context.
    If you doesnt know the answer or cant be found in context simply say abra ka dabra and move one.
                                         
    <context> {context} </context>
                                         
    Here's the question: {input}
''')
    
    llm = st.session_state.llm
    doc_chain = create_stuff_documents_chain(llm, template)
    retriever = st.session_state.retriever
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    res = retrieval_chain.invoke({"input": prompt})
    return res['answer']

st.title("RAG with GROQ & LLAMA")

prompt = st.text_input(label="Ask anything related to magic")

if st.button("Prepare document"):
   pre_process_docs()
   st.write("Documents ready!")

if prompt:
    answer = query_db_and_llm(prompt=prompt)
    st.write(str(answer))