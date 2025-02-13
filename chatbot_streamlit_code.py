#IMPORTING LIBRARIES FOR USING PRETRAINED MODEL 
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

#LOAD THE PRETRAINED MODEL AND TOKENIZER
model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#STREAMLIT APP TITLE
st.title("🤖 AI Chatbot with BlenderBot")
st.markdown("Type your message below and chat with the AI!")

#SIDEBAR TITLE FOR OLD MESSAGES
st.sidebar.title("📜 Chat History")

#INITIALIZE THE CHAT HISTORY
if "messages" not in st.session_state:
    st.session_state.messages = []

#DISPLAY THE OLD MESSAGE IN SIDEBAR
for msg in st.session_state.messages:
    st.sidebar.write(f"**{msg['role'].capitalize()}**: {msg['content']}")

#CREATING THE USER INPUT
user_input = st.chat_input("Type your message...")

#STORE THE USER MESSAGE
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    #DISPLAY THE USER MESSAGE IN MAIN CHAT 
    with st.chat_message("user"):
        st.write(user_input)

    #MAKE THE CHATBOT RESPONSE
    inputs = tokenizer(user_input, return_tensors="pt")
    reply_ids = model.generate(**inputs)
    response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

    #STORING THE CHATBOT MESSAGE 
    st.session_state.messages.append({"role": "assistant", "content": response})

    #DISPLAY THE CHATBOT MESSAGE IN MAIN BOT
    with st.chat_message("assistant"):
        st.write(response)

