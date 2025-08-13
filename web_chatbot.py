import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from search import RAG_pipeline
from PIL import Image

im = Image.open("assets/logo.png")
# init streamlit app
st.set_page_config(page_title="Chatbot CSE UPI", page_icon=im)

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

# Ask anything about Computer Science UPI
user_question = st.chat_input("Tanyakan apa saja tentang Ilmu Komputer UPI")

if user_question:
    # get chat history (if available)
    chat_history = st.session_state.messages
    # add user message to chat history and display it
    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))

    with st.chat_message("assistant"):
        # Placeholder untuk menampilkan streaming response
        response_container = st.empty()
        full_response = ""
        
        # Call RAG_pipeline and get the streaming response
        stream = RAG_pipeline(query=user_question, chat_history=chat_history)
        
        # Stream the response using st.write_stream
        for chunk in stream:
            full_response += chunk.content
            response_container.markdown(full_response)
        
        # Add final AI message to chat history
        st.session_state.messages.append(AIMessage(full_response))