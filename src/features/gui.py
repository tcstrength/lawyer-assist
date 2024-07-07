# -*- coding: utf-8 -*-
"""
Author: tcstrength
Date: 2024-07-04
"""

import streamlit as st 

if __name__ == "__main__":
    st.title("Sentiment Analysis Web Application")
    text_input = st.text_area("Enter text for analysis:")
    if st.button("Analyze"):
        result = {
            "sentiment": "positive",
            "score": "0.9"
        }
        st.write(f"Sentiment: {result['sentiment']}")
        st.write(f"Score: {result['score']}")