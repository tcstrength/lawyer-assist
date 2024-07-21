# -*- coding: utf-8 -*-
"""
%cd ../
Author: tcstrength
Date: 2024-07-04
"""

import streamlit as st
from gui import api

st.title("Echo Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Nhập câu hỏi hoặc tình huống bạn đang gặp phải?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        for response in api.post_chat_stream(prompt):
            st.markdown(response)
        st.markdown("Bạn có câu hỏi nào khác?")
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})