from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ('system','you are a helpful assistant please respond the quires'),
        ('user' , 'Question:{question}')
    ]
)

# streamlit
st.title('Langchain Chatapp')
input_text = st.text_input('Search the topic you want')

# ollama llm
llm = Ollama(model = 'llama2')
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))