import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings

st.set_page_config("Q&A")
st.header("Q&A With Astra DB")
user_input = st.text_input("Enter Your Question Here")

# HuggingFace LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="conversational"
)
model = ChatHuggingFace(llm=llm)

# Functions
def get_pdf_docs(files):
    all_docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
            print(tmp_path)
        
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

def get_astra_db_index(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    ASTRA_DB_APPLICATION_TOKEN = ""
    ASTRA_DB_ID = ""
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

    astra_vector = Cassandra(
        embedding=embeddings,
        table_name="QnA",
        session=None,
        keyspace=None
    )
    astra_vector.add_documents(chunks)

    return VectorStoreIndexWrapper(vectorstore=astra_vector)

# Sidebar for file upload
with st.sidebar:
    st.title('Your Document')
    files = st.file_uploader("Enter Your PDF Here", accept_multiple_files=True)

    if st.button("Process"):
        if files:
            with st.spinner("Processing..."):
                docs = get_pdf_docs(files)
                st.success(f"{len(docs)} total docs loaded.")

                chunks = get_text_chunks(docs)
                st.success(f"{len(chunks)} chunks generated.")

                vector_index = get_astra_db_index(chunks)
                st.session_state.vector_index = vector_index  # Save to session state

                st.success("Your data is ready for Q&A!")
        else:
            st.warning("Please upload at least one PDF file.")

# Q&A Section
if user_input:
    if "vector_index" in st.session_state:
        with st.spinner("Thinking..."):
            retriever = st.session_state.vector_index.vectorstore.as_retriever()
            relevant_docs = retriever.similarity_search(user_input,k=3)
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
