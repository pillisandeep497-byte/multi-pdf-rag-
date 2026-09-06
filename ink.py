from langchain_openrouter import ChatOpenRouter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv 
import os 
from langchain_community.vectorstores import FAISS
import streamlit as st
load_dotenv()


st.title("MULTI-PDF-RAG-BOT")
pdf=[
    "RAG/sample3.pdf",
    "RAG/sample4.pdf"
]

documents=[]

for p in pdf:
    loader=PyPDFLoader(p)
    doc=loader.load()

    for j in doc:
        j.metadata["source"]=p

    documents.extend(doc)


splitter= RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunk=splitter.split_documents(documents)
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vectore_store = FAISS.from_documents(
    chunk,

    embedding
    
)

retrievar=vectore_store.as_retriever(
    search_kwargs={"k":5}

)

prompt = ChatPromptTemplate.from_template(
    """
You are a PDF assistant.

Answer ONLY from the provided context.
If the answer is not found in the context, say:
"I could not find that information in the uploaded PDFs."

Context:
{context}

Question:
{input}
"""
)

llm=ChatOpenRouter(
    model="gpt-oss-20b",
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0

)

document_chain=create_stuff_documents_chain(
    llm,
    prompt
)
retrievar_chain=create_retrieval_chain(
    retrievar,
    document_chain
)
if "messages" not in st.session_state:
    st.session_state.messages=[]



for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input=st.chat_input("ask your questions ")

if user_input:
        st.session_state.messages.append(
                {"role":"user","content":user_input}
    
            )
        with st.chat_message("user"):
             st.markdown(user_input)


        with st.chat_message("assistant"):
             with st.spinner("finding..."):
    


                response=retrievar_chain.invoke(
            {"input":user_input}
                )
                

                answer=response["answer"]

                st.session_state.messages.append(
            {"role":"assistant","content":answer}

            )
                with st.chat_message("assistant"):
                    st.markdown(answer)
