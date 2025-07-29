from langchain_core.tools import tool
import os
from langchain_ollama.chat_models import ChatOllama

@tool
def list_dir():
    '''Returns the list of directories from point where this program is being ran from'''
    dirs = os.listdir()
    return dirs

def tool_call_helper(call_info: dict):
    args = call_info['args']
    match call_info['name']:
        case "list_dir":
            return list_dir.invoke(call_info)
        case _:
            return f"No method found for tool: {call_info['name']}"

tools = [list_dir]
llm = ChatOllama(model="llama3.1:8b")
llm_with_tools = llm.bind_tools(tools)

res = llm_with_tools.invoke("Hey can you list out the directories please?")
for call in res.tool_calls:
    print(tool_call_helper(call).content)