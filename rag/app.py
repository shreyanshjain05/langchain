import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
import bs4

# Load environment variables
load_dotenv()
os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')

# Load text document (uncomment if needed)
# loader = TextLoader('speech.txt')
# text_document = loader.load()

# Web-based document loader
# web_loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-title", "post-content", "post-header")
#         )
#     )
# )
# web_documents = web_loader.load()
# print(web_documents)

# PDF document loader and text splitter
pdf_loader = PyPDFLoader('attention.pdf')
pdf_docs = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
pdf_documents = text_splitter.split_documents(pdf_docs)
# print(pdf_documents[:5])

# Vector embedding using Ollama and Chroma
db_chroma = Chroma.from_documents(
    pdf_documents[:20],
    OllamaEmbeddings(model="llama3.2")
)

# Chroma vector database query
query_chroma = 'Who all are the authors of Attention is All You Need research paper?'
result_chroma = db_chroma.similarity_search(query_chroma)
print("Chroma Result:\n", result_chroma[0].page_content)

# FAISS vector database
db_faiss = FAISS.from_documents(
    pdf_documents[:20],
    OllamaEmbeddings(model="llama3.2")
)

# FAISS vector database query
query_faiss = 'What is an attention mechanism?'
result_faiss = db_faiss.similarity_search(query_faiss)
print("FAISS Result:\n", result_faiss[0].page_content)
