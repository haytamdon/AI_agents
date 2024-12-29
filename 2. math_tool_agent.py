from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

class MessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built 
    pass

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

if __name__ == "__main__":

    llm_name = "qwen2.5:0.5b"
    llm = ChatOllama(model=llm_name)

    llm_with_tools = llm.bind_tools([multiply])

    builder = StateGraph(MessagesState)
    builder.add_node("tool_calling_llm", tool_calling_llm)
    builder.add_node("tools", ToolNode([multiply]))
    builder.add_edge(START, "tool_calling_llm")
    builder.add_conditional_edges(
        "tool_calling_llm",
        tools_condition,
    )
    builder.add_edge("tools", END)
    graph = builder.compile()

    messages = graph.invoke({"messages": HumanMessage(content="Multiply 10 and 5")})
    for m in messages['messages']:
        m.pretty_print()