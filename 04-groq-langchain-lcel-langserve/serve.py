# Example of how we can directly expose chains or runnables as endpoints
from fastapi.responses import RedirectResponse
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.output_parsers import StrOutputParser

llm = OllamaLLM(
    model="llama3.1:8b"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sarcastic senior developer with over 15 years of experience who specialises in roasting."),
    ("user", "Tell me something about {ask}")
])

parser = StrOutputParser()

chain = prompt | llm | parser

# Using langserve and fastapi to expose
app = FastAPI(title="RoastSmith", version="0.0.1")
add_routes(app, chain, path='/roast')

# Run using: fastapi run serve.py from current proj. dir