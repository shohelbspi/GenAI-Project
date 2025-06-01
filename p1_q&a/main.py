import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

st.set_page_config("Q&A")
st.header("Q&A With Your PDFs")
user_input = st.text_input("Enter Your Question Here")

# Load HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)

# === FUNCTIONS ===
def get_pdf_docs(files):
    all_docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            print(tmp)
            tmp.write(file.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        all_docs.extend(docs)
    return all_docs

def get_text_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

def get_faiss_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index = FAISS.from_documents(chunks, embedding=embeddings)
    return VectorStoreIndexWrapper(vectorstore=faiss_index)

# === SIDEBAR FILE UPLOAD ===
with st.sidebar:
    st.title('Your Document')
    files = st.file_uploader("Upload PDF(s)", type=['pdf'], accept_multiple_files=True)

    if st.button("Process"):
        if files:
            with st.spinner("Processing..."):
                docs = get_pdf_docs(files)
                st.success(f"{len(docs)} total docs loaded.")

                chunks = get_text_chunks(docs)
                st.success(f"{len(chunks)} chunks created.")

                vector_index = get_faiss_index(chunks)
                st.session_state.vector_index = vector_index

                st.success("✅ Your data is ready for Q&A!")
        else:
            st.warning("Please upload at least one PDF file.")

# === Q&A Section ===
if user_input:
    if "vector_index" in st.session_state:
        with st.spinner("Thinking..."):
            retriever = st.session_state.vector_index.vectorstore.as_retriever()
            relevant_docs = retriever.get_relevant_documents(user_input)
            context = "\n\n".join(doc.page_content for doc in relevant_docs)

            prompt = f"""Answer the question based on the following context:
            {context}

            Question: {user_input}
            Answer:"""

            response = model.invoke(prompt)
            st.subheader("Answer")
            st.write(response.content)
    else:
        st.warning("Please upload and process a PDF first.")
