# -*- coding: utf-8 -*-
"""
%cd ../
Author: tcstrength
Date: 2024-07-04
"""

import streamlit as st
from gui import api
from gui.api import ModelArch

st.title("Hỗ trợ pháp luật lĩnh vực Tiền tệ ngân hàng".upper())
st.subheader("MSSV: 23C01024 - 23C01025 - 23C01037")

model_option = st.selectbox(
    label="__Chọn mô hình:__",
    options=(ModelArch.FINETUNE.value, ModelArch.RAG.value)
)

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
            model=model_option)
        for response in stream_response:
            full_text += response
            stream_placeholder.markdown(full_text)
        stream_placeholder.markdown(full_text + f"(`Trả lời bằng {model_option}`)")
    # Add assistant response to chat history
    st.session_state.messages += [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": full_text}
    ]