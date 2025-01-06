import streamlit as st
from bs4 import BeautifulSoup
import base64
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.vectorstores import Qdrant
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from dotenv import load_dotenv
import hashlib

load_dotenv()

# Set GROQ API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize Qdrant client
client = QdrantClient(
    url="http://localhost:6333",  # Default Qdrant server address
)

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

compressor = FlashrankRerank()

contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

system_prompt = """You're a helpful AI assistant. Given a user question and some Machine Lecture notes, answer the user question and provide citations. If none of the articles answer the question, just say you don't know.

Remember, you must return both an answer and citations. A citation consists of a VERBATIM quote that justifies the answer and the ID of the quote article. Return a citation for every quote across all articles that justify the answer. Use the following format for your final output:

<cited_answer>
    <answer></answer>
    <citations>
        <citation><source_id></source_id><quote></quote></citation>
        <citation><source_id></source_id><quote></quote></citation>
        ...
    </citations>
</cited_answer>

Here are the lecture notes:{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

def get_collection_name(pdf_files):
    """Generate a unique collection name based on the PDF files"""
    pdf_names = sorted([f.name for f in pdf_files])
    combined_names = "_".join(pdf_names)
    return f"pdf_{hashlib.md5(combined_names.encode()).hexdigest()[:10]}"

def create_collection(collection_name, vector_size):
    """Create a new Qdrant collection if it doesn't exist"""
    collections = client.get_collections().collections
    existing_collections = [c.name for c in collections]
    
    if collection_name in existing_collections:
        # Delete existing collection if it exists
        client.delete_collection(collection_name)
    
    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )

def initialize_rag_chain():
    """Initialize or reinitialize the RAG chain with current PDFs"""
    pdf_files = st.session_state.get('pdf_files', [])
    collection_name = get_collection_name(pdf_files)
    
    all_docs = []
    for pdf_file in pdf_files:
        # Save temporary file
        with open(f"temp_{pdf_file.name}", "wb") as f:
            f.write(pdf_file.getvalue())
        
        # Load and process the PDF
        loader = PyPDFLoader(f"temp_{pdf_file.name}")
        documents = loader.load()
        
        # Clean up temporary file
        os.remove(f"temp_{pdf_file.name}")
        
        all_docs.extend(documents)
    
    combined_text = " ".join(doc.page_content for doc in all_docs)
    text_splitter = SemanticChunker(embeddings)
    docs = text_splitter.create_documents([combined_text])
    
    sample_embedding = embeddings.embed_query("sample text")
    vector_size = len(sample_embedding)
    
    create_collection(collection_name, vector_size)
    
    qdrant = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
    )
    
    # Add documents to the collection
    qdrant.add_documents(docs)
    
    retriever = qdrant.as_retriever()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=retriever
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        compression_retriever, 
        contextualize_q_prompt
    )
    
    question_answer_chain = create_stuff_documents_chain(
        llm, 
        qa_prompt
    )
    
    return create_retrieval_chain(
        history_aware_retriever, 
        question_answer_chain
    )

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_pdfs" not in st.session_state:
    st.session_state.current_pdfs = set()

st.set_page_config(page_title="Multi-PDF Chat App", layout="wide")

# Sidebar for PDF upload
st.sidebar.title("Upload PDFs")
pdf_files = st.sidebar.file_uploader(
    "Upload PDF files", 
    type=["pdf"],
    accept_multiple_files=True
)

if pdf_files:
    st.session_state.pdf_files = pdf_files
    
    # Check if PDFs have changed
    current_pdf_names = {f.name for f in pdf_files}
    if current_pdf_names != st.session_state.current_pdfs:
        st.session_state.current_pdfs = current_pdf_names
        if 'rag_chain' in st.session_state:
            del st.session_state.rag_chain
        
        st.session_state.rag_chain = initialize_rag_chain()
    
    st.sidebar.markdown("## Uploaded PDFs")
    for pdf_file in pdf_files:
        with open(f"temp_{pdf_file.name}", "wb") as f:
            f.write(pdf_file.getvalue())
        
        try:
            with open(f"temp_{pdf_file.name}", "rb") as f:
                pdf_bytes = f.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="300" height="300" type="application/pdf"></iframe>'
            
            st.sidebar.markdown(f"### {pdf_file.name}")
            st.sidebar.markdown(pdf_display, unsafe_allow_html=True)
            
            os.remove(f"temp_{pdf_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error displaying PDF {pdf_file.name}: {e}")

st.title("Chat with your PDFs")

if pdf_files:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                try:
                    soup = BeautifulSoup(message["content"], 'xml')
                    answer = soup.find('answer').get_text()
                    citations_tags = soup.find_all('citation')
                    
                    st.markdown("### 📝 Answer")
                    st.markdown(f"_{answer}_")
                    
                    with st.expander("📚 View Supporting Citations"):
                        for idx, citation_tag in enumerate(citations_tags, 1):
                            source_id = citation_tag.find('source_id').get_text()
                            quote = citation_tag.find('quote').get_text()
                            st.markdown(f"""
                            ---
                            **Citation #{idx}**
                            - 📍 **Source:** `{source_id}`
                            - 💡 **Quote:** "{quote}"
                            """)
                except Exception as e:
                    st.write(message["content"])
            else:
                st.write(message["content"])

    user_input = st.chat_input("Type your message here")

    if user_input and 'rag_chain' in st.session_state:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        response = st.session_state.rag_chain.invoke({
            "input": user_input,
            "chat_history": st.session_state.chat_history
        })
        answer_str = response.get("answer", "I don't know.")

        try:
            soup = BeautifulSoup(answer_str, 'xml')
            answer = soup.find('answer').get_text()
            citations_tags = soup.find_all('citation')

            citations = []
            for citation_tag in citations_tags:
                source_id = citation_tag.find('source_id').get_text()
                quote = citation_tag.find('quote').get_text()
                citations.append((source_id, quote))

            with st.chat_message("assistant"):
                st.markdown("### 📝 Answer")
                st.markdown(f"_{answer}_")
                
                with st.expander("📚 View Supporting Citations"):
                    for idx, (source_id, quote) in enumerate(citations, 1):
                        st.markdown(f"""
                        ---
                        **Citation #{idx}**
                        - 📍 **Source:** `{source_id}`
                        - 💡 **Quote:** "{quote}"
                        """)

        except Exception as e:
            st.write(answer_str)
            st.error("An error occurred while parsing the response.")

        st.session_state.chat_history.append({"role": "assistant", "content": answer_str})