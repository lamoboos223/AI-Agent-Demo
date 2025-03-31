import streamlit as st
from agent import agent, HumanMessage, ToolMessage


def remove_status_code_message(messages):
    """Remove ToolMessage with content '200'."""
    messages["messages"] = [
        msg
        for msg in messages["messages"]
        if not (isinstance(msg, ToolMessage) and msg.content == "200")
    ]
    return messages


# Usage: $ streamlit run src/agent_example/app.py
st.title("Service Call Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.write(message.content)

# Chat input
if prompt := st.chat_input("What would you like me to do?"):
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            messages = agent.invoke({"messages": [HumanMessage(content=prompt)]})

            # Display each message from the agent
            print(messages["messages"])

            messages = remove_status_code_message(messages)
            for m in messages["messages"]:
                st.write(m.content)
                st.session_state.messages.append(m)
