import asyncio
import random  # temp
import time  # temp

import streamlit as st

from src.helper.st_response_handler import get_response_generator
from src.workflow.orchestrator import get_default_workflow


async def main():
    """
    This is the main Streamlit app that runs the chatbot.
    """
    st.session_state.orchestrator_workflow = get_default_workflow()
    st.title("Budget 2024 Chatbot")

    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []
    for message in st.session_state.display_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def clear_chat():
        st.session_state.display_messages.clear()
        st.session_state.chat_history.clear()

    if prompt := st.chat_input(
        "Hello! Feel free to ask me anything about Budget 2024."
    ):
        st.session_state.display_messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            full_prompt = ""
            for chat in st.session_state.chat_history:
                full_prompt += f"User: {chat['user']}\nAssistant: {chat['assistant']}\n"
            full_prompt += f"User: {prompt}\n"
            print(full_prompt)
            message_placeholder = st.empty()
            response = ""
            response_generator = get_response_generator(
                full_prompt, st.session_state.orchestrator_workflow
            )
            async for chunk in response_generator:
                chunk = chunk.replace("$", "\$")
                chunk = chunk.replace("\\\$", "\$")
                response += chunk
                message_placeholder.markdown(response + "▌")
            message_placeholder.markdown(response)

        st.session_state.display_messages.append(
            {"role": "assistant", "content": response}
        )

        st.session_state.chat_history.append({"user": prompt, "assistant": response})
        if len(st.session_state.chat_history) > 5:
            st.session_state.chat_history.pop(0)

        st.button("Clear chat", on_click=lambda: clear_chat())
        st.info(
            """
Pro Tips:\n
- For follow-up questions, keep your chat history so that the bot can understand the full context.
- For new questions, clear your chat history to help the bot focus on your latest query.
"""
        )


if __name__ == "__main__":
    asyncio.run(main())
