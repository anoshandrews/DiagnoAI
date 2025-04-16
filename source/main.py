import os
import json
import torch
from x_ray_prompt import prompts

import streamlit as st
from groq import Groq
from PIL import Image
import torchvision.transforms as transforms

from transformers import AutoImageProcessor, AutoModelForImageClassification

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

model_id = 'khengkok/vit-medical'

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)

class_names = model.config.id2label

#initializing the chat history if streamlit session state is not available yet
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def preprocess_image(file):
    """
    Preprocesses a JPEG or PNG image for model inference.

    Args:
        file (UploadedFile): A file-like object containing the image (e.g., from Streamlit upload).

    Returns:
        torch.Tensor: A tensor of shape [1, 3, 224, 224] ready for input to a CNN.
    """
    image = Image.open(file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def dummy_cnn_output():
    return """
    ✅ Image received and processed.
    
    🧠 AI Analysis (placeholder):
    - No visible abnormalities.
    - Good contrast and resolution.
    - Ready for further diagnostic evaluation.
    """



#just initilizing the function for now for the button ---code to be added here later
def generate_report():
        pass

# streamlit page title
col1, col2 = st.columns([4,1])  # Adjust width ratio as needed

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
    Performs inference on a single medical image using a ViT-based model.

    Args:
        image_file: Path to the image file.

    Returns:
        tuple: (predicted_class, class_probs) where predicted_class is the label index (int),
               and class_probs is a list of softmax probabilities for each class.
    """

    image = Image.open(image_file).convert("RGB")

    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze()

        predicted_class_index = probs.argmax().item()

        predicted_class_name = class_names[predicted_class_index]

    return predicted_class_name

# with col2:
#         st.write('')
#         st.write('')
#         st.write('')
#         st.write('')
#         if col2.button("Create Report", key = 'report_button'):
#                 with st.spinner("Generating your diagnosis report..."):
#                         generate_report()
#                 st.success("Done! You can now download your report.")

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

As DiagnoAI, explain in simple language what this condition means. Then continue with your usual symptom triage — ask the patient for symptoms, duration, severity, etc., like you normally do.

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