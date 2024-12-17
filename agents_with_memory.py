from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import MemorySaver

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

if __name__ == "__main__":
    # define model
    llm_name = "llama3.2"
    llm = ChatOllama(model=llm_name)

    # State the tools
    tools = [add, multiply, divide]
    # Bind the tools to the llm
    llm_with_tools = llm.bind_tools(tools)

    # Model System prompt
    sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetics on a set of inputs.")

    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    memory = MemorySaver()


    react_graph = builder.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "1"}}

    while True:
        user_input=input("User: ")
        if user_input.lower() in ["quit","q"]:
            print("Good Bye")
            break
        messages = [HumanMessage(content=user_input)]
        messages = react_graph.invoke({"messages": messages}, config)
        for m in messages['messages']:
            m.pretty_print()