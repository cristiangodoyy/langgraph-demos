from dotenv import load_dotenv
from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage


load_dotenv()


# 1 I define tools
@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    print(f'Usando la tools: triple')
    return float(num) * 3

@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    print(f'Usando la tools: add')
    return a + b 

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    print(f'Usando la tools: Subtraction')
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    print(f'Usando la tools: multiply')
    return a * b

# 2
tools = [TavilySearch(), triple, add, subtract, multiply]  # distintas tool que va a usar el Agente

# 3
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)

# 4 Node agent_reason
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    SYSYEM_MESSAGE="""
    You are a helpful assistant that can use tools to answer questions.
    """
    response = llm.invoke([{"role": "system", "content": SYSYEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}

# 5 Node tool
#tool_node = ToolNode(tools)


def main():
    
    # ReAct: Reasoning + Acting
    AGENT_REASON="agent_reason"
    ACT= "act"  # actor
    LAST = -1

    def should_continue(state: MessagesState) -> str:
        if not state["messages"][LAST].tool_calls:
            return END
        return ACT

    flow = StateGraph(MessagesState)
    flow.add_node(AGENT_REASON, run_agent_reasoning)
    flow.set_entry_point(AGENT_REASON)
    flow.add_node(ACT, ToolNode(tools))  # defino el nodo tool

    flow.add_conditional_edges(
        source=AGENT_REASON,
        path=should_continue,
        path_map={END:END, ACT:ACT}
    )

    flow.add_edge(ACT, AGENT_REASON)

    app = flow.compile()
    app.get_graph().draw_mermaid_png(output_file_path="flow.png")

    print("Hello ReAct LangGraph with Function Calling")
    
    # dependiendo del content el agente sabe que tool usar
    res = app.invoke({
        "messages": [
            HumanMessage(
                #content="What is the temperature in Rosario, Santa Fe, Argentina? List it and then triple it."
                content="Add 40 + 12 and then multiply the result by 6."            
                #content="search for 3 job postings for an blockchain engineer using langchain in the Rosario, Santa Fe, Argentina area on linkedin and list their details."
                )
            ]})

    print(res["messages"][LAST].content)


if __name__ == "__main__":
    main()
