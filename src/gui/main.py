# -*- coding: utf-8 -*-
"""
%cd ../
Author: tcstrength
Date: 2024-07-04
"""

import streamlit as st
from gui import api
from gui.api import ModelArch

st.title("23C01024 - 23C01025 - 23C01037")

# Function to handle button click
def set_active_button(model):
    st.session_state.selected_button = model.value
    
if 'selected_button' not in st.session_state:
    set_active_button(ModelArch.FINETUNE)

col1, col2 = st.columns(2)

with col1:
    if st.button(ModelArch.FINETUNE.value, key="btn1"):
        set_active_button(ModelArch.FINETUNE)

with col2:
    if st.button(ModelArch.RAG.value, key="btn2"):
        set_active_button(ModelArch.RAG)

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

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        stream_placeholder = st.empty()
        full_text = ""
        stream_response = api.post_chat_stream(
            user_input=prompt,
            history=st.session_state.messages,
            model=st.session_state.selected_button)
        for response in stream_response:
            full_text += response
            stream_placeholder.markdown(full_text)

        st.markdown("Bạn có câu hỏi nào khác?")
    # Add assistant response to chat history
    st.session_state.messages += [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": full_text}
    ]