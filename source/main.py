import os
import json
import torch
from x_ray_prompt import prompts
import numpy as np
from datetime import datetime 

import streamlit as st
from groq import Groq
from PIL import Image
import torchvision.transforms as transforms
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer

#streamlit page configuration

report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding_model = embedding_model.to('cpu')

working_dir = os.path.dirname(os.path.abspath(__file__))

vectorstore_path = os.path.join(working_dir,'vectorstore')

vectorstore = FAISS.load_local(vectorstore_path, embeddings = embedding_model, allow_dangerous_deserialization= True)


os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
        page_title = 'DiagnoAI',
        page_icon = '⚕️',
        layout = 'centered'
        )

config_data = json.load(open(f"{working_dir}/config.json"))

GROQ_API_KEY = config_data['GROQ_API_KEY']

#save the environment variable and Groq API key

os.environ['GROQ_API_KEY'] = GROQ_API_KEY

client = Groq()

model_id = 'Salesforce/blip-image-captioning-base'

model = BlipForConditionalGeneration.from_pretrained(model_id)

processor = BlipProcessor.from_pretrained(model_id, use_fast = True)


#initializing the chat history if streamlit session state is not available yet
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_pubmed_docs(query):
    abstracts = pubmed_api_call(query)
    docs = [Document(page_content = a) for a in abstracts]
    return docs

def store_in_vector_db(docs):
    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(docs, embedding = embeddings, allow_dangerous_deserialization = True)
    vectorstore.save_local(vectorstore_path)
    return vectorstore

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def summarize_chat(chat_history):
        user_inputs = "\n".join([msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"])

        summary_prompt = f"""Based on the following conversation, sumamrize the user's reported symptoms in the structured format:
        
        {user_inputs}
            
        Respond with a clear summary of:
        - Symptom description
        - Duration
        - Severity
        - Triggers
        - Associated symptoms
        """

        summary_response = client.chat.completions.create(
            model = 'llama-3.1-8b-instant',
            messages = [
                {"role": "system", "content": "You are a medical assistant summarizing patient symptom inputs."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        
        summary = summary_response.choices[0].message.content.strip()
        return summary

def retrieve_context(summary):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(summary, k = 3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context

def build_report(symptom_summary, medical_context):
    final_prompt = f""" Generate a medical report using the symptom summary and relevant medical knowledge below.
    Symptom Summary:
    {symptom_summary}

    Medical Knowledge:
    {medical_context}

    Format the report with:
    - Patient Summary alond with date and time {report_time}
    - Possible Conditions
    - Recommendations
    """
    report_response = client.chat.completions.create(
        model = 'llama-3.1-8b-instant',
        messages = [
            {"role": "system", "content": "You are a doctor writing a preliminary patient diagnosis report."},
            {"role": "user", "content": final_prompt}
        ]
    )
    return report_response.choices[0].message.content.strip()

def generate_downloadable_report(content, filename = 'diagnosis_report.txt'):
    with open(filename, 'w') as f:
        f.write(content)

    with open(filename, 'rb') as file:
        st.download_button('Download Report', file, file_name = filename)

def generate_report():
    with st.spinner('Generating report....'):
        summary = summarize_chat(st.session_state.chat_history)

        context = retrieve_context(summary)

        report = build_report(summary, context)

        generate_downloadable_report(report)

        st.success('Report generated!')


# streamlit page title
col1, col2 = st.columns([4,2])  # Adjust width ratio as needed

with col1:
    st.markdown(
    """
    <h1 style='
        background: linear-gradient(to right, #FF3C38, #FFB347);
        -webkit-background-clip: text;
        color: transparent;
        font-size: 4rem;
        text-align: center;
        padding-bottom: 0.5em;
    '>⚕️DiagnoAI</h1>
    """,
    unsafe_allow_html=True
)

def run_inference(image_file):
    """
    Analyzes an input image using a vision-language model to assess whether 
    the visual condition appears medically serious or self-resolving.

    This function takes a user-uploaded image (e.g., of a wound, rash, or injury),
    passes it through a BLIP or similar image captioning model along with a 
    triage-style prompt, and returns a natural language response suggesting whether 
    medical attention might be necessary.

    Parameters:
        image_file (str or file-like object): Path to the image file or 
            a file object representing the image to be analyzed.

    Returns:
        str: The model's textual response evaluating the condition in the image.
            Typically includes recommendations like whether to visit a doctor.

    Example:
        >>> response = run_inference("rash_photo.jpg")
        >>> print(response)
        "This appears to be a minor skin irritation and may heal on its own, 
        but if symptoms worsen, consult a doctor."
    """
    image = Image.open(image_file).convert("RGB")
    
    prompt = (
        "Describe the condition shown in this image. "
        "Does this look medically serious, or is it something that will heal on its own? "
        "Should the person visit a doctor?"
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    answer = processor.decode(output[0], skip_special_tokens = True)
    # print(answer)
    return answer


with col2:
        st.write('')
        st.write('')
        st.write('')
        if col2.button("Create Report", key = 'report_button'):
                generate_report()

# initial assistant GREETING
greeting = """
*Please describe your Symptoms or upload an Image so I can assist you with an AI-powered analysis*"""

# st.session_state.chat_history.append({"role": "assistant", "content": greeting})
with st.chat_message('assistant'):
    st.markdown(greeting)

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


#input field for user message
user_prompt = st.chat_input(
        'Tell me about your symptoms....', 
        accept_file = True,                     #for taking in images as well
        file_type = ['jpg','png','jpeg'] )      #formats specified

if user_prompt:
    user_text = user_prompt.text
    user_files = user_prompt.files

    if user_text:
        st.session_state.chat_history.append({"role" : "user", "content" : user_text})
        with st.chat_message('user'):
            st.markdown(user_text)
        messages = [
            {"role": "system", 
            "content": """"You are a professional medical assistant by the name DiagnoAI trained in symptom triage and patient interview. 
Your goal is to collect a patient's symptoms in a structured and efficient way.

Start by greeting the user and asking for general symptoms. Then proceed with follow-up questions 
based on what the user shares. Be friendly but focused. Prioritize clarity and detail.

DO NOT give any diagnosis, possible conditions, or medical advice. 
Only collect information that would help a doctor later. 
Ask follow-up questions like:
- Duration
- Severity
- Location
- Triggers or relieving factors
- Associated symptoms (fever, nausea, etc.)

Keep the conversation going until the user explicitly says they have no more symptoms to share,
or it becomes clear there's no new info to gather.
 Finish with a message like: 'Thank you. I've collected enough information to generate your report.'" """},
            *st.session_state.chat_history
        ]

        response = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=messages
        )

        assistant_response = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
        with st.chat_message('assistant'):
            st.markdown(assistant_response)

    if user_files:
        for file in user_files:
            # Save file or process it with CNN here
            st.chat_message("user").markdown(f"Uploaded image: `{file.name}`")
            # Optional: preview
            st.image(file)
            # Steps to be added to pass the image to CNN:Resnet-50
            with st.spinner('Analysing image....'):
                result = run_inference(file)
        # st.chat_message('assistant').markdown(result)
        
        image_prompt = f"""The AI image classifier detected the condition as **{result}**.

As DiagnoAI, explain in simple language what output you received to the user. Then continue with your usual symptom triage — ask the patient for symptoms, duration, severity, etc., like you normally do.

Avoid giving medical advice or diagnosis. Only collect structured information a doctor can use later.
"""

            # Compose message list
        image_chat = [
            {"role": "system", 
            "content": """You are named DiagnoAI trained in patient triage.
You explain results like to a human with compassion and ask structured questions about symptoms. Collect relevant details."""},
            {"role": "user", "content": image_prompt}
        ]

        image_response = client.chat.completions.create(
            model='llama-3.1-8b-instant',
            messages=image_chat
        )

        followup = image_response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": followup})
        with st.chat_message('assistant'):
            st.markdown(followup)