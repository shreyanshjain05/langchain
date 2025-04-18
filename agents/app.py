from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_ollama.llms import OllamaLLM
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain import hub

# Set up the Wikipedia query tool
tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=5, doc_content_chars_max=4000))

# Load documents from a website
loader = WebBaseLoader("https://docs.smith.langchain.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

# Set up the FAISS vector store for document retrieval
vectordb = FAISS.from_documents(documents, OllamaEmbeddings(model='llama3.2'))
retriever = vectordb.as_retriever()

# Set up the retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    "langsmith_search",
    "Search for information about LangSmith. For any questions about LangSmith, you must use this tool!"
)

# Define the prompt for the agent (can be customized)
prompt = hub.pull("hwchase17/openai-functions-agent")

# Set up the language model
model = OllamaLLM(model='llama3.2')

# Create the agent using the tools and prompt
agent = create_tool_calling_agent(model, [tool, retriever_tool], prompt)

# Set up the agent executor
agent_executor = AgentExecutor(agent=agent, tools=[tool, retriever_tool], verbose=True)

# Print the agent executor information
print(agent_executor)


# Function to run the agent with a query
def run_agent_with_query(query):
    try:
        result = agent_executor.invoke({"input": query})
        return result
    except Exception as e:
        return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    # Example queries to test the agent
    queries = [
        "What is LangSmith and how can it help with LLM applications?",
        "Tell me about the history of artificial intelligence.",
        "What features does LangSmith offer for debugging?"
    ]

