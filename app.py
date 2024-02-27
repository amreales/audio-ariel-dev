import streamlit as st
import requests
import os
import assemblyai as aai

# Set page config
st.set_page_config(
   page_title='Ariel Transcripciones',
   layout='wide',  # Use 'wide' for expanded margins
   initial_sidebar_state='auto',  # Can be 'auto', 'expanded', 'collapsed'
   menu_items={
       'Get Help': 'https://www.extremelyhelpfulmenuitem.com',
       'Report a bug': "https://www.bugreportpage.com",
       'About': "# This is a header. This is an *extremely* cool app!"
   }
)

# Replace with your AssemblyAI API key
aai.settings.api_key = os.environ.get("ASSEMBLYAI_API_KEY")

def transcribe_audio(file):
    # Language code for Spanish
    language_code = "es"

    # Send audio file to AssemblyAI for transcription using binary data
    headers = {
        'authorization': aai.settings.api_key,
        'Content-Type': 'application/octet-stream'
    }
    
    # Convert Streamlit UploadedFile to bytes for direct binary upload
    file_bytes = file.getvalue()

    response = requests.post(
        'https://api.assemblyai.com/v2/upload',
        headers=headers,
        data=file_bytes  # Send file as binary data
    )
    file_url = response.json()['upload_url']

    # Enable speaker labels for speaker diarization
    config = aai.TranscriptionConfig(language_code=language_code, speaker_labels=True)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_url, config=config)

    return transcript

# Streamlit UI customizations
def local_css():
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Use the function above to apply custom CSS
local_css()

# Set custom font to Segoe UI Web
custom_css = """
<style>
    html, body, [class*="st-"] {
        font-family: "Segoe UI Web", sans-serif;
    }
</style>
"""

# Hide the "Made with Streamlit" footer
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.markdown(custom_css, unsafe_allow_html=True)

# Streamlit UI
st.image('logofull.png', width=200)  # Adjust the width as needed
st.title('Transcripciones de audio')

# File uploader
uploaded_file = st.file_uploader("Sube un archivo de audio para transcribirlo y resumirlo rápidamente. Tu información permanecerá privada y no se almacenará.", type=['mp3', 'wav', 'mpeg', 'mp4', 'm4a'])

if uploaded_file is not None:
    # Display a message while file is being transcribed
    with st.spinner('Transcribiendo...'):
        transcript = transcribe_audio(uploaded_file)

        if transcript.status == aai.TranscriptStatus.error:
            st.error('Error durante la transcripción: ' + transcript.error)
        else:
            # Display the transcription and summary
            formatted_text = "\n".join(
                f"Interlocutor {utterance.speaker}: {utterance.text}\n" for utterance in transcript.utterances
            )
            summary_prompt = "Haz un resumen detallado de la transcripción"
            summary_result = transcript.lemur.task(summary_prompt)
            summary = summary_result.response

            st.subheader('Resumen')
            st.text_area('Resumen general del audio suministrado', summary, height=300)

            st.subheader('Transcripción')
            st.text_area('Transcripción completa del audio original', formatted_text, height=400)

