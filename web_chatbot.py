import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from search import RAG_pipeline
from PIL import Image

im = Image.open("logo.png")
# init streamlit app
st.set_page_config(page_title="RAG Chatbot CSE UPI", page_icon=im)

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

user_question = st.chat_input("Ask anything about Computer Science UPI")

if user_question:
    # get chat history (if available)
    chat_history = st.session_state.messages
    # add user message to chat history and display it
    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))

    # get response from RAG
    ai_message = RAG_pipeline(query=user_question, chat_history=chat_history)

    # add AI message to chat history and display it
    with st.chat_message("assistant"):
        st.markdown(ai_message)

        st.session_state.messages.append(AIMessage(ai_message))