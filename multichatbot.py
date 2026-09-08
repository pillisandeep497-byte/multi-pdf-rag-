from langchain_openrouter import ChatOpenRouter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# ----------------------------------
# Streamlit Config
# ----------------------------------
st.set_page_config(
    page_title="Multi PDF RAG Bot",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Multi PDF RAG Bot")

# ----------------------------------
# API Key
# ----------------------------------
api_key = st.secrets["OPENROUTER_API_KEY"]

# ----------------------------------
# Create Vector Store
# ----------------------------------
@st.cache_resource
def create_vector_store():

    pdf_files = [
        "RAG/sample3.pdf",
        "RAG/sample4.pdf"
    ]

    documents = []

    for pdf in pdf_files:

        loader = PyPDFLoader(pdf)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = pdf

        documents.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store


vector_store = create_vector_store()

# ----------------------------------
# Retriever
# ----------------------------------
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# ----------------------------------
# Prompt
# ----------------------------------
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful PDF assistant.

Answer ONLY from the provided context.

If the answer is not available in the context, respond with:

"I could not find that information in the uploaded PDFs."

Context:
{context}

Question:
{input}
"""
)

# ----------------------------------
# LLM
# ----------------------------------
llm = ChatOpenRouter(
    model="gpt-oss-20b",
    openrouter_api_key=api_key,
    temperature=0
)

# ----------------------------------
# Chains
# ----------------------------------
document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

retrieval_chain = create_retrieval_chain(
    retriever,
    document_chain
)

# ----------------------------------
# Chat History
# ----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------------
# User Input
# ----------------------------------
user_input = st.chat_input(
    "Ask questions about your PDFs..."
)

if user_input:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Searching PDFs..."):

            response = retrieval_chain.invoke(
                {"input": user_input}
            )

            answer = response["answer"]

            # Get sources
            sources = set()

            if "context" in response:
                for doc in response["context"]:
                    sources.add(
                        doc.metadata.get(
                            "source",
                            "Unknown Source"
                        )
                    )

            if sources:
                answer += "\n\n### Sources\n"
                for source in sources:
                    answer += f"- {source}\n"

            st.markdown(answer)

            # Store assistant message
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )
