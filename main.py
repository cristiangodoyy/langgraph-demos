from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from chains import revisor, first_responder
from tool_executor import execute_tools


if __name__ == "__main__":
    MAX_ITERATIONS = 2
    def draft_node(state: MessagesState):
        """Draft the initial response."""
        response = first_responder.invoke({"messages": state["messages"]})
        return {"messages": [response]}


    def revise_node(state: MessagesState):
        """Revise the answer based on tool results."""
        response = revisor.invoke({"messages": state["messages"]})
        return {"messages": [response]}


    def event_loop(state: MessagesState) -> Literal["execute_tools", END]:
        """Determine whether to continue or end based on iteration count."""
        count_tool_visits = sum(
            isinstance(item, ToolMessage) for item in state["messages"]
        )
        num_iterations = count_tool_visits
        if num_iterations > MAX_ITERATIONS:
            return END
        return "execute_tools"


    builder = StateGraph(MessagesState)
    builder.add_node("draft", draft_node)
    builder.add_node("execute_tools", execute_tools)
    builder.add_node("revise", revise_node)
    builder.add_edge(START, "draft")
    builder.add_edge("draft", "execute_tools")
    builder.add_edge("execute_tools", "revise")
    builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
    graph = builder.compile()

    print(graph.get_graph().draw_mermaid())

    res = graph.invoke(
        {
            "messages": [
                # {
                #     "role": "user",
                #     "content": "Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.",
                # }

                # {
                #     "role": "user",
                #     "content": "Escribe sobre los problemas de los motores de reservas de hoteles."
                #                 " Dime cuales son los principales problemas que tienen las motores de reserva con respecto a la busqueda"
                #                 " inicial de itinerarios y con respecto a las politicas de cancelaciones de cada reserva."
                # }
                
                # {
                #     "role": "user",
                #     "content": "Escribe sobre los destinos de playa que eligen los Argentinos en Brasil."
                #                 " Dime cuales son los principales destinos de playa que eligen los Argentinos en Brasil en marzo de 2026."
                # }
                
                # {
                #     "role": "user",
                #     "content": "Escribe sobre los problemas mas comunes que tienen los desarrolladores de aplicaciones cuando tiene que eleigir los servicios de los proveedores: AWS, AZURE, GCP."
                #                 " Dime cuales son los principales problemas con respecto a despliegues y que es lo mas costoso para los desarrolladores."
                # }
                
                # {
                #     "role": "user",
                #     "content": "Escribe sobre lo que significa adoptar IaC y herramientas multi-cloud (Terraform, Pulumi) puede reducir los costes operativos y acelerar despliegues en AWS, AZURE O GCP"
                #                 " Dime cuales son los dime cuales son los principales problemas que tiene la IaC y las herramientas multi-cloud para los desarrolladores."
                # }
                
                # {
                #     "role": "user",
                #     "content": "Escribe sobre las diferencias de las herramientas multi-cloud: Terraform, Pulumi, Heroku, Vercel."
                #                 " Dime cuales son los principales problemas que tienen los desarrolladores al intentar utilizar estas herramientas."
                # }
                
                {
                    "role": "user",
                    "content": "Escribe sobre las herramientas de IaC y multi-cloud que existen para reducir los costes operativos y acelerar despliegues en AWS, AZURE O GCP."
                                " Dime cuales son las principales y las mas utilizadas por los desarrolladores y empresas."
                }
            ]
        }
    )

    # Extract the final answer from the last message with tool calls
    last_message = res["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print(last_message.tool_calls[0]["args"]["answer"])
    print(res)

