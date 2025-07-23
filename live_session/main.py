# Imports 
import whisper
import sounddevice as sd
import numpy as np
import os
from api_keys import OPENAI_API_KEY, MURF_API_KEY
from openai import OpenAI
from murf import Murf

def create_openai_client(api_key):
    return OpenAI(api_key=api_key)

def create_murf_client(api_key):
    return Murf(api_key=api_key)


def get_openai_response(client, user_input, instructions, model="gpt-4o"):
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=user_input   
    )
    return response.output_text

def print_voices():
    voices = create_murf_client(MURF_API_KEY).text_to_speech.get_voices()
    for voice in voices:
        print(f'VOICE ID: {voice.voice_id}, Name:{voice.display_name}, MOODS:{voice.available_styles}')

def stream_text_to_speech(client, text, output_path):
    try:
        print(f"Streaming text: {text}")
        audio_stream = client.text_to_speech.stream(
            text=text,
            voice_id="en-US-ariana",
            format="MP3",
            style="Conversational",
            pitch=0
        )
        
        with open(output_path,"wb") as file:
            for chunk in audio_stream:
                file.write(chunk)
                
        print(f'Audio file created at: {output_path}')
        
    except Exception as e:
        print(f'ERROR:{e}')
        raise
        


class Message:
    def __init__(self, user_name:str, text:str, message_type:str):
        self.user_name = user_name
        self.text = text 
        self.message_type = message_type

def record_audio(model):
    """Record audio and return transcribed text using Whisper"""
    try:
        # Audio recording parameters
        sample_rate = 16000
        duration = 5  # seconds
        
        print("Listening... (speak now)")
        
        # Record audio
        audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype=np.float32)
        sd.wait()  # Wait until recording is finished
        
        # Convert to the format Whisper expects
        audio_data = audio_data.flatten().astype(np.float32)
        
        # Transcribe with Whisper
        result = model.transcribe(audio_data)
        text = result["text"].strip()
        
        if text:
            print(f"Heard: {text}")
            return text
        else:
            print("No speech detected")
            return None
            
    except Exception as e:
        print(f'Error with speech recognition: {e}')
        return None


def play_audio(file_path):
    """Play audio file using system default player and wait for completion"""
    try:
        import platform
        import subprocess
        system = platform.system()
        
        if system == "Darwin":  # macOS
            subprocess.run(["afplay", file_path], check=True)
        elif system == "Linux":
            subprocess.run(["aplay", file_path], check=True)
        elif system == "Windows":
            subprocess.run(["start", "/wait", file_path], shell=True, check=True)
        else:
            print(f"Audio saved to: {file_path}")
    except Exception as e:
        print(f"Error playing audio: {e}")


def print_message(message: Message):
    """Print a message with appropriate formatting"""
    if message.message_type == "system":
        print(f"\n {message.user_name}: {message.text}")
    elif message.message_type == "ai":
        print(f"\n{message.user_name}: {message.text}")
    else:
        print(f"\n{message.user_name}: {message.text}")

def main():
    print(" Voice chat - just speak!")
    print("=" * 50)
    
    # Initialize clients
    openai_client = create_openai_client(OPENAI_API_KEY)
    murf_client = create_murf_client(MURF_API_KEY)
    
    # Load Whisper model (this will download on first run)
    print("Loading model...")
    whisper_model = whisper.load_model("base")
    print("Model loaded!")
    
    chat_history = []
    
    while True:
        try:
            # Always use voice input
            user_input = record_audio(whisper_model)
            
            if not user_input:
                print(" Speak again...")
                continue
            
            # Add user message to history and display
            user_message = Message("You", user_input, "user")
            chat_history.append(user_message)
            print_message(user_message)
            
            # Get AI response
            instructions = "You are a helpful assistant. Provide short and overly sarcastic, slightly offensive responses."
            ai_response = get_openai_response(openai_client, user_input, instructions)
            
            # Add AI message to history and display
            ai_message = Message("AI", ai_response, "ai")
            chat_history.append(ai_message)
            print_message(ai_message)
            
            # Convert to speech and play
            try:
                file_path = os.path.abspath("audio.mp3")
                stream_text_to_speech(murf_client, ai_response, file_path)
                
                # Play audio and wait for completion
                play_audio(file_path)
                
            except Exception as e:
                print(f"Audio Error: {e}")
                
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()    