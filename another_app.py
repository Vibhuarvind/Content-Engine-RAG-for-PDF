import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import os
import pickle
from transformers import pipeline

# Set the page configuration
st.set_page_config(page_title="Vidisha Assignment", layout="wide")

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text, chunk_size=10000):
    text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return text_chunks

# Function to create and save a FAISS vector store
def get_vector_store(text_chunks, model, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    embeddings = model.encode(text_chunks, batch_size=8, show_progress_bar=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(save_dir, "faiss_index"))
    with open(os.path.join(save_dir, "text_chunks.pkl"), "wb") as f:
        pickle.dump(text_chunks, f)

# Function to load the FAISS vector store
def load_vector_store(save_dir):
    index = faiss.read_index(os.path.join(save_dir, "faiss_index"))
    with open(os.path.join(save_dir, "text_chunks.pkl"), "rb") as f:
        text_chunks = pickle.load(f)
    return index, text_chunks

# Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
save_dir = "C:\\Users\\hp\\Documents\\faiss_data"  # Replace with your directory

# Initialize conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

# Sidebar for file upload
with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
    if st.button("Submit & Process", key="process_button"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks, model, save_dir)
            st.success("Done")

# Function to handle the user message and get a response
def handle_message(message):
    index, text_chunks = load_vector_store(save_dir)
    query_embedding = model.encode([message])
    D, I = index.search(query_embedding, k=5)  # Adjust k as needed
    retrieved_text = " ".join([text_chunks[i] for i in I[0]])

    qa_pipeline = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad")
    response = qa_pipeline(question=message, context=retrieved_text)
    return response['answer']

# Display the conversation
st.markdown("## Chat with your PDF")
for msg in st.session_state.conversation:
    st.markdown(f"**You:** {msg['user']}")
    st.markdown(f"**Bot:** {msg['bot']}")

# User input
user_input = st.text_input("Type your question here...", key="user_input")

if st.button("Send", key="send_button"):
    if user_input:
        bot_response = handle_message(user_input)
        st.session_state.conversation.append({'user': user_input, 'bot': bot_response})
        st.experimental_rerun()
