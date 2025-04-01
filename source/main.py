import os
import json

import streamlit as st
from groq import Groq

#streamlit page configuration

st.set_page_config(
        page_title = 'DiagnoAI',
        page_icon = '⚕️',
        layout = 'centered'
        )

working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))

GROQ_API_KEY = config_data['GROQ_API_KEY']

#save the environment variable and Groq API key

os.environ['GROQ_API_KEY'] = GROQ_API_KEY

client = Groq()

#initializing the chat history if streamlit session state is not available yet

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# streamlit page title

st.title('⚕️ DignoAI')

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


#input field for user message

user_prompt = st.chat_input('Tell me about your symptoms....')

if user_prompt:
    st.chat_message('user').markdown(user_prompt)
    st.session_state.chat_history.append({"role" : "user", "content" : user_prompt})

    # providing instructions to pass to GROQ

    messages = [
            {"role" : "system" , "content" : "You are a medical assistant, your task is to diagnose the patient's disease. Your task is to diagnose the symptoms and do nothing else, do not say anything about the disease, just ask questions further and further to get the most details until the point when the user cant list more symptos"},
            *st.session_state.chat_history
            ]

    response = client.chat.completions.create(
            model = 'llama-3.1-8b-instant',
            messages = messages
            )
    
    assistant_response = response.choices[0].message.content
    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
    
    #displaying the LLM's response

    with st.chat_message("assistant"):
        st.markdown(assistant_response)




