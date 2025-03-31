from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from typing import Literal
import os
import pandas as pd
import requests
import json
from graphviz import Digraph


os.environ["GROQ_API_KEY"] = "******"


# Define tools
@tool
def make_http_call(service_name: str) -> int:
    """Make a http call to the service.

    Args:
        service_name: name of the service to call from services.csv
    """
    df = pd.read_csv("./src/services.csv", skipinitialspace=True)
    matching_services = df[df["service_name"] == service_name]
    if matching_services.empty:
        available_services = df["service_name"].tolist()
        return f"Error: Service '{service_name}' not found. Available services are: {available_services}"

    service_row = matching_services.iloc[0]
    service_url = service_row["service_url"]
    response = requests.post(service_url, service_row["service_payload"])
    return response.status_code, "response.json()"


# Augment the LLM with tools
tools = [make_http_call]
tools_by_name = {tool.name: tool for tool in tools}
llm = ChatGroq(model="deepseek-r1-distill-llama-70b")
llm_with_tools = llm.bind_tools(tools)


# Nodes
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    df = pd.read_csv("./src/services.csv", skipinitialspace=True)
    services_list = "\n".join(
        [f"- {row['service_name']}: {row['description']}" for _, row in df.iterrows()]
    )

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content=f"""You are a helpful assistant tasked with performing http calls to a set of services.
                                    Available services are:
                                    {services_list}

                                    Always choose the most specific and appropriate service for the situation."""
                    )
                ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: dict):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        service_name = tool_call["args"].get("service_name", "unknown service")
        status_code, response_json = tool.invoke(tool_call["args"])
        result.append(
            ToolMessage(
                content="##### Calling Service:",
                tool_call_id=tool_call["id"],
            )
        )
        result.append(
            ToolMessage(content=f"**{service_name}**", tool_call_id=tool_call["id"])
        )
        result.append(ToolMessage(content=status_code, tool_call_id=tool_call["id"]))
        result.append(
            ToolMessage(content="##### Response:", tool_call_id=tool_call["id"])
        )
        result.append(
            ToolMessage(content=f"{response_json}", tool_call_id=tool_call["id"])
        )

    return {"messages": result}


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["tool_node", "end"]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call and we haven't succeeded yet, continue
    if last_message.tool_calls:
        return "Action"
    # Otherwise, stop
    return END


def check_status_and_continue(state: MessagesState) -> Literal["tool_node", "end"]:
    """Check the status code in messages and decide whether to continue or end."""
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            try:
                content = json.loads(msg.content)
                if content == 200:
                    return END
            except json.JSONDecodeError:
                continue
    return "Action"


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "Action": "tool_node",
        END: END,
    },
)

# Add conditional edge from tool_node to either end or llm_call
agent_builder.add_conditional_edges(
    "tool_node",
    check_status_and_continue,
    {END: END, "Action": "llm_call"},
)


# Add visualization before compilation
def visualize_agent_graph(state_graph):
    dot = Digraph(comment="Agent Workflow")

    # Configure graph attributes
    dot.attr(rankdir="LR")  # Left to right layout
    dot.attr("node", shape="rectangle", style="rounded")

    # Add nodes with styling
    dot.node("START", "START", shape="oval", color="green")
    dot.node("END", "END", shape="oval", color="red")
    dot.node("llm_call", "LLM Call", color="blue")
    dot.node("tool_node", "Tool Node", color="purple")

    # Add edges with labels
    dot.edge("START", "llm_call")
    dot.edge("llm_call", "tool_node", "Action")
    dot.edge("llm_call", "END", "End")
    dot.edge("tool_node", "llm_call", "Action")
    dot.edge("tool_node", "END", "End")

    # Save the visualization
    dot.render("agent_workflow", format="png", cleanup=True)
    print("Graph visualization saved as 'agent_workflow.png'")


visualize_agent_graph(agent_builder)

# Compile the agent
agent = agent_builder.compile()
