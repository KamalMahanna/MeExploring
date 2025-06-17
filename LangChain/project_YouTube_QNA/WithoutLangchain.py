import streamlit as st

st.title("YouTube Q&A")


with st.sidebar:
    youtube_link = st.text_input("YouTube video Link")
    if st.button("Submit"):
        