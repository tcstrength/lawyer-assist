# -*- coding: utf-8 -*-
"""
%cd ../
Author: tcstrength
Date: 2024-07-04
"""

import streamlit as st
from gui.api import get_gpu_status, post_chat_stream, EmbeddingModel, get_relevant_documents, get_model


# Function to display chat interface
def chat_interface():
    st.title("Chat Interface")

    # Embedding selection
    embedding_model = st.selectbox("Select Embedding Model", EmbeddingModel._member_names_)
    
    # Chat input
    user_input = st.text_input("Your message:", key="chat_input")
    st.write(f"Câu hỏi: {user_input}")
    if user_input:
        stream_placeholder = st.empty()
        full_text = ""
        stream_response = post_chat_stream(user_input, embedding_model)
        for response in stream_response:
            full_text += response
            stream_placeholder.markdown(full_text)

        # st.write(response)

# Function to display retrieval results
def retrieval_results():
    st.title("Retrieval Results")
    
    query = st.text_input("Enter query to retrieve results:")
    
    if query:
        st.write(f"Retrieving results:")

        col1, col2 = st.columns(2)
        with col1:
            AllMiniLML6v2_result = get_relevant_documents(query, limit=10, model_id=EmbeddingModel.AllMiniLML6v2.value)

            # Display results
            st.subheader(f"Results from {EmbeddingModel.AllMiniLML6v2.name}")
            for result in AllMiniLML6v2_result:
                st.write(result)
        
        with col2:
            halong_results = get_relevant_documents(query, limit=10, model_id=EmbeddingModel.HALONG.value)
            
            st.subheader(f"Results from {EmbeddingModel.HALONG.name}")
            for result in halong_results:
                st.write(result)

# Main application logic
def main():
    st.title("Hỗ trợ pháp luật lĩnh vực Tiền tệ ngân hàng".upper())
    st.subheader("MSSV: 23C01024 - 23C01025 - 23C01037")

    st.sidebar.title("Menu")
    menu_option = st.sidebar.selectbox("Select Page", ["Chat Interface", "Retrieval Results"])
    
    if menu_option == "Chat Interface":
        chat_interface()
    elif menu_option == "Retrieval Results":
        retrieval_results()
    
    model = get_model()
    st.sidebar.write(f"Model: {model}")

    # Fetch GPU status and display in the sidebar
    gpu_status = get_gpu_status()
    gpu_available = "Yes" if gpu_status["gpu_available"] else "No"
    gpu_name = gpu_status["gpu_name"]

    st.sidebar.subheader("GPU Info")
    st.sidebar.write(f"GPU Available: {gpu_available}")
    st.sidebar.write(f"GPU Name: {gpu_name}")

if __name__ == "__main__":
    main()