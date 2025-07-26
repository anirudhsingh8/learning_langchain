import streamlit as st
from llm_service import LLMService
import ollama

service = LLMService()
st.title("LLM Chat app")
st.sidebar.title("Configure your llm")
res = ollama.list()
available_models = map(lambda x: x.model, res.models)

model = st.sidebar.selectbox(label="Choose model", options=available_models)

query = st.text_input(label="Enter a message")
print(query)
if query:
    st.write(service.get_completion(query, model))
else:
    st.write("Please add a query")