import streamlit as st
import speech_recognition as sr
import pyttsx3
from transformers import T5ForConditionalGeneration, T5Tokenizer, CLIPProcessor, CLIPModel
import threading
import time
import openai 
from PIL import Image

engine = pyttsx3.init()
engine.setProperty('rate', 150) 
engine.setProperty('volume', 0.9) 
model_name = "t5-small" 
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
openai.api_key = 'API KEY '  # Replace with your API key

def speak_text_thread(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    
    threading.Thread(target=run).start()
    
def recognize_speech():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        st.info("🎤 Listening... Speak now.")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return "Listening timed out. Please try again."
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError as e:
            return f"Could not request results; {e}"

def get_chatgpt_response(user_message):
    try:
        response = openai.Completion.create(
            model="gpt-3.5-turbo", 
            prompt=user_message,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error with OpenAI API: {e}"


def get_clip_response(image):
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    image_features = clip_model.get_image_features(**inputs)
    return image_features

def add_custom_css():
    st.markdown(
        """
        <style>
        .main-header {
            color: #4CAF50;
            font-size: 42px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
            font-family: 'Arial', sans-serif;
        }
        .textbox-icon {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .textbox {
            width: 80%;
            border: 2px solid #FF6347;
            border-radius: 15px;
            padding: 15px;
            font-size: 18px;
            background-color: #F0E68C;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease-in-out;
        }
        .textbox:hover {
            border-color: #FF4500;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        .icon-button {
            width: 60px;
            height: 60px;
            background-color: #FF6347;
            color: white;
            border: none;
            border-radius: 50%;
            font-size: 30px;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
        }
        .icon-button:hover {
            background-color: #FF4500;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            transform: translateY(-3px);
        }
        .message-bubble {
            padding: 15px;
            border-radius: 25px;
            max-width: 60%;
            margin: 10px 0;
            font-size: 16px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            font-family: 'Verdana', sans-serif;
        }
        .user-message {
            background-color: #E0F7FA;
            align-self: flex-end;
            border-radius: 25px 25px 0 25px;
        }
        .bot-message {
            background-color: #C8E6C9;
            align-self: flex-start;
            border-radius: 25px 25px 25px 0;
        }
        .message-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .message-container {
            overflow-y: auto;
            max-height: 500px;
            padding: 10px;
            background-color: #FAF9F6;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def simulate_typing(text, speed=0.1):
    for char in text:
        st.write(char, end="", flush=True)
        time.sleep(speed)
    st.write("") 

def main():
    add_custom_css()
    st.markdown('<h1 class="main-header">🤖 ChatGPT Voice Assistant</h1>', unsafe_allow_html=True)
    st.write("Interact with the chatbot using **voice**, **text**, or **image** input!")
    if "listening" not in st.session_state:
        st.session_state.listening = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "text_input" not in st.session_state:
        st.session_state.text_input = ""
    if len(st.session_state.chat_history) == 0:  
        welcome_message = "Hi, how can I assist you today?"
        st.session_state.chat_history.append({"role": "bot", "message": welcome_message})
        speak_text_thread(welcome_message)
    col1, col2 = st.columns([4, 1])
    with col1:
        text_input_placeholder = "Type your message here..."
        if "text_input" in st.session_state:
            user_input = st.text_input("", placeholder=text_input_placeholder, key="text_input")
        else:
            user_input = st.text_input("", placeholder=text_input_placeholder, key="text_input")
    with col2:
        if st.button("🎙️", key="microphone", help="Click to start voice input", use_container_width=True):
            st.session_state.listening = True  # Activate listening mode
            st.session_state.chat_history.append({"role": "user", "message": "🎤 Listening..."})
            st.session_state.chat_history.append({"role": "bot", "message": "🎤 Listening... Please speak!"})
            with st.spinner('Listening...'):
                user_input = recognize_speech()
                if user_input:
                    st.session_state.chat_history.append({"role": "user", "message": user_input})
                    response = get_chatgpt_response(user_input)
                    st.session_state.chat_history.append({"role": "bot", "message": response})
                    speak_text_thread(response) 
    
    st.session_state.listening = False
    if st.button("Send", key="send"):
        if user_input:
            st.session_state.chat_history.append({"role": "user", "message": user_input})
            response = get_chatgpt_response(user_input)
            st.session_state.chat_history.append({"role": "bot", "message": response})
            speak_text_thread(response) 

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True) 
        clip_response = get_clip_response(image)
        st.session_state.chat_history.append({"role": "user", "message": "Image uploaded for analysis."})
        st.session_state.chat_history.append({"role": "bot", "message": f"Image processed. Response from CLIP: {clip_response}"})
        speak_text_thread("I have processed the image.") 
    st.markdown("<div class='message-container'>", unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(
                f"<div class='message-bubble user-message'>{chat['message']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='message-bubble bot-message'>{chat['message']}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
if __name__ == "__main__":
    main()
