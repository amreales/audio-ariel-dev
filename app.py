import streamlit as st
import requests
import os
import assemblyai as aai

# Set page config
st.set_page_config(
    page_title='DEV',
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

def ask_question(transcribed_text, question):
    # Create a Lemur task to answer a question based on the transcribed text
    lemur_prompt = f"{transcribed_text}\n\nPregunta: {question}\nRespuesta:"
    lemur_response = transcript.lemur.task(lemur_prompt, model="davinci")
    return lemur_response.response

# Streamlit UI customizations and CSS functions remain unchanged
# ...

# Streamlit UI
st.image('logofull.png', width=200)
st.title('Transcripciones de audio')

# Create columns for Transcript, Summary, and Chat
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Transcripción')
    uploaded_file = st.file_uploader("Sube un archivo de audio para transcribirlo y resumirlo rápidamente. Tu información permanecerá privada y no se almacenará.", type=['mp3', 'wav', 'mpeg', 'mp4', 'm4a'])
    if uploaded_file is not None:
        with st.spinner('Transcribiendo...'):
            transcript = transcribe_audio(uploaded_file)

            if transcript.status == aai.TranscriptStatus.error:
                st.error('Error durante la transcripción: ' + transcript.error)
            else:
                formatted_text = "\n".join(
                    f"Interlocutor {utterance.speaker}: {utterance.text}\n" for utterance in transcript.utterances
                )
                st.text_area('Transcripción completa del audio original', formatted_text, height=400)

with col2:
    st.subheader('Resumen')
    if uploaded_file is not None and transcript.status != aai.TranscriptStatus.error:
        summary_prompt = "Haz un resumen detallado de la transcripción"
        summary_result = transcript.lemur.task(summary_prompt)
        summary = summary_result.response
        st.text_area('Resumen general del audio suministrado', summary, height=300)

with col3:
    st.subheader('Chat con el texto')
    if uploaded_file is not None and transcript.status != aai.TranscriptStatus.error:
        user_question = st.text_input("Haz una pregunta basada en la transcripción:")
        if user_question:
            with st.spinner('Buscando respuesta...'):
                answer = ask_question(formatted_text, user_question)
                st.text_area('Respuesta', answer, height=300)
