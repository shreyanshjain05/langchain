from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Load PDF document
pdf_path = "attention.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Increased for better context retrieval
    chunk_overlap=100,  # Avoids loss of important context
    length_function=len,
    is_separator_regex=False,
)

documents = text_splitter.split_documents(docs)

# Create FAISS vector store with embeddings
embedding_model = OllamaEmbeddings(model="llama3.2")
db = FAISS.from_documents(documents, embedding_model)

# Define retriever
retriever = db.as_retriever()

# Initialize LLM
llm = Ollama(model="llama3.2")

# Define chat prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based on the provided context.
Think step by step before providing a detailed answer.

<context>
{context}
</context>

Question: {input}
""")

# Create document processing chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Query for retrieval
query = "The Transformer follows this overall architecture using stacked self-attention and point-wise"

# Retrieve and generate response
response = retrieval_chain.invoke({"input": query})

# Print final response
print("\n Answer:")
print(response.get("answer", "No relevant answer found."))