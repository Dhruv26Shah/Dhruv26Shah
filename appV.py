import streamlit as st
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage
import pytesseract
from gtts import gTTS
import io
import base64
import logging
from api_setup import get_google_api_key

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure Google API Key
try:
    GOOGLE_API_KEY = get_google_api_key()  # Ensure the API key is being fetched properly
except Exception as e:
    st.error(f"Failed to load API key: {e}")
    st.stop()

# Initialize models through LangChain
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
vision_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Error handling function
def handle_error(error):
    logging.error(error)
    st.error(f"Error: {str(error)}")

# Helper function to process images
def prepare_image(image):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    return image_bytes.getvalue()

# Scene understanding function
def scene_understanding(image):
    try:
        image_bytes = prepare_image(image)
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """As an AI assistant for visually impaired individuals, provide a detailed description of this image."""
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
                }
            ]
        )
        response = vision_llm.invoke([message])
        return response.content
    except Exception as e:
        handle_error(e)

# Object detection function
def detect_objects_and_obstacles(image):
    try:
        image_bytes = prepare_image(image)
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """Analyze this image for safety and navigation purposes."""
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64.b64encode(image_bytes).decode()}"
                }
            ]
        )
        response = vision_llm.invoke([message])
        return response.content
    except Exception as e:
        handle_error(e)

# Text extraction function
def extract_and_process_text(image):
    try:
        extracted_text = pytesseract.image_to_string(image)
        if not extracted_text.strip():
            return "No text detected in the image."
        template = "Enhance this extracted text for clarity:\nTEXT: {text}"
        prompt = PromptTemplate(input_variables=["text"], template=template)
        chain = LLMChain(llm=llm, prompt=prompt)
        return chain.run(text=extracted_text)
    except Exception as e:
        handle_error(e)

# Text-to-speech function
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang="en")
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp.getvalue()
    except Exception as e:
        handle_error(e)

# Main Streamlit App
def main():
    st.set_page_config(page_title="Vision Assistant", layout="wide")
    st.title("SeeForMe : AI Assistant for Visually Impaired")

    st.sidebar.title("About SeeForMe")
    st.sidebar.info(
        "AI Assistant for Visually Impaired:\n"
        "• Scene Descriptions\n"
        "• Text Reading\n"
        "• Object Detection\n"
        "• Task-Specific Guidance"
    )

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        feature = st.radio("Select Feature", ["Scene Description", "Text Reading", "Object Detection"])

        if feature == "Scene Description" and st.button("Analyze Scene"):
            with st.spinner("Analyzing the scene..."):
                result = scene_understanding(image)
                st.write(result)
                st.audio(text_to_speech(result), format="audio/mp3")

        elif feature == "Text Reading" and st.button("Extract Text"):
            with st.spinner("Extracting text..."):
                result = extract_and_process_text(image)
                st.write(result)
                st.audio(text_to_speech(result), format="audio/mp3")

        elif feature == "Object Detection" and st.button("Detect Objects"):
            with st.spinner("Detecting objects..."):
                result = detect_objects_and_obstacles(image)
                st.write(result)
                st.audio(text_to_speech(result), format="audio/mp3")

if __name__ == "__main__":
    main()
