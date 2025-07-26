from langchain_ollama import OllamaLLM
from langchain_core.messages import SystemMessage, HumanMessage

class LLMService:
    def get_completion(self, prompt: str, model: str) -> str:
        llm = OllamaLLM(model=model)
        res = llm.invoke(prompt)
        
        return res
