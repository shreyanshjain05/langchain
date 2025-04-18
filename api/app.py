from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
load_dotenv()

os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')

app = FastAPI(title='Langchain Server',
              version = '1.0',
              description='API server'
              )
add_routes(
    app,
    ChatGoogleGenerativeAI(model = 'gemini-2.0-flash' ,google_api_key=os.getenv('GEMINI_API_KEY')),
    path = '/gemini'
)
model = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash' ,google_api_key=os.getenv('GEMINI_API_KEY'))

llm = Ollama(model = 'llama2')
prompt1 = ChatPromptTemplate.from_template("Write a essay on {topic} in 100 words")
prompt2 = ChatPromptTemplate.from_template("Write a poem on {topic} in 100 words")

add_routes(
    app,
    prompt1 | model,
    path = '/essay'
)

add_routes(
    app,
    prompt2|model,
    path = '/poem'
)

if __name__ == "__main__":
    uvicorn.run(app,host="localhost" , port=8000)
