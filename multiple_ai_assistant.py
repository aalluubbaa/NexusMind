import os
import glob
import json
from datetime import datetime
import threading
import wave
import winsound
import time
import cv2
import pyaudio
import numpy as np
import pyperclip
import requests
import io
from openai import OpenAI
from PIL import ImageGrab, Image
from groq import Groq
import google.generativeai as genai
import whisperx
import torch
import gc
import keyboard
import anthropic
from deepgram import DeepgramClient, SpeakOptions
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
from sentence_transformers import SentenceTransformer
from pydub import AudioSegment

from api_keys import GROQ_API_KEY, GENAI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPGRAM_API_KEY

# Global variables and configurations
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_OUTPUT_FILENAME = "output.wav"
MIN_RECORDING_LENGTH = 0.5
COOLDOWN_PERIOD = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16"  # change to "int8" if low on GPU memory
batch_size = 2  # reduce if low on GPU memory

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GENAI_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
DEEPGRAM_API_KEY = DEEPGRAM_API_KEY

# WhisperX model initialization
whisperx_model = None

# Webcam initialization
web_cam = cv2.VideoCapture(0)

MODELS = {
    '1': {
        'name': 'Groq (Gemma2-9b-it)',
        'client': groq_client,
        'model': 'gemma2-9b-it'
    },
    '2': {
        'name': 'Groq (Llama-3-70b)',
        'client': groq_client,
        'model': 'llama3-70b-8192'
    },
    '3': {
        'name': 'Claude 3.5 Sonnet',
        'client': anthropic_client,
        'model': 'claude-3-sonnet-20240229'
    }
}

selected_model = None
memory = None

sys_msg = (
    'You are a multi-modal AI voice assistant. You have a built-in advanced memory system'
    'Use the information to make a more personal conversation when necessary'
    'Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses extremely word-efficient, clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]

generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 1024
}

safety_settings = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)

is_speaking = False
is_recording = False
frames = []
recording_thread = None
last_recording_time = 0
recording_lock = threading.Lock()

class AdvancedMemory:
    def __init__(self, file_path='advanced_memory.json', model_name='all-MiniLM-L6-v2'):
        self.file_path = file_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name).to(self.device)
        self.memories = self.load_memories()

    def load_memories(self):
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
                for memory in data:
                    memory['embedding'] = np.array(memory['embedding'])
                return data
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def save_memories(self):
        serializable_memories = []
        for memory in self.memories:
            serializable_memory = memory.copy()
            serializable_memory['embedding'] = memory['embedding'].tolist()
            serializable_memories.append(serializable_memory)
        
        with open(self.file_path, 'w') as f:
            json.dump(serializable_memories, f, indent=2)

    @torch.no_grad()
    def add_memory(self, content, importance=1):
        embedding = self.model.encode([content], device=self.device, convert_to_numpy=True)[0]
        memory = {
            'content': content,
            'importance': importance,
            'timestamp': datetime.now().isoformat(),
            'embedding': embedding
        }
        self.memories.append(memory)
        self.save_memories()

    @torch.no_grad()
    def get_relevant_memories(self, query, limit=5):
        query_embedding = self.model.encode([query], device=self.device, convert_to_numpy=True)[0]
        
        memory_embeddings = np.array([memory['embedding'] for memory in self.memories])
        
        similarities = np.dot(memory_embeddings, query_embedding) / (
            np.linalg.norm(memory_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[-limit:][::-1]
        
        return [self.memories[i] for i in top_indices]

    def forget_old_memories(self, max_memories=100):
        if len(self.memories) > max_memories:
            self.memories.sort(key=lambda x: (x['importance'], x['timestamp']), reverse=True)
            self.memories = self.memories[:max_memories]
            self.save_memories()

    def get_memory_summary(self, limit=10):
        recent_memories = sorted(self.memories, key=lambda x: x['timestamp'], reverse=True)[:limit]
        return "\n".join([f"- {memory['content']}" for memory in recent_memories])

    def update_memory_importance(self, content, new_importance):
        for memory in self.memories:
            if memory['content'] == content:
                memory['importance'] = new_importance
                self.save_memories()
                return True
        return False

def select_model():
    global selected_model
    print("Available models:")
    for key, value in MODELS.items():
        print(f"- {key}: {value['name']}")
    
    while True:
        choice = input("Select a model: ").lower()
        if choice in MODELS:
            selected_model = MODELS[choice]
            return selected_model
        else:
            print("Invalid choice. Please try again.")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""
def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return prompt, sources

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are the vision analysis AI that provides semantic meaning from images to provide context '
        'to send to another AI that will create a response to the user. Do not respond as the AI assistant '
        'to the user. Instead take the user prompt input and try to analyze the image efficiently, focusing on elements relevant to the user prompt. '
        'Provide a concise summary (less than 10 sentences) covering key visual information. '
        'Include main subjects, actions, text content, and relevant context. '
        'Prioritize details that directly relate to the user query. '
        'Avoid unnecessary repetition and overly specific details unless crucial. '
        f'assistant who will respond to the user. USER PROMPT: {prompt}'
    )
    response = model.generate_content([prompt, img])
    print("Gemini Response:", response.text)
    return response.text

def ai_prompt(prompt, img_context, model_info):
    if img_context:
        prompt += f'\n\nIMAGE CONTEXT: {img_context}'
    
    messages = [{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': prompt}]
    
    if model_info['name'].startswith('Groq'):
        chat_completion = model_info['client'].chat.completions.create(
            messages=messages,
            model=model_info['model']
        )
        response = chat_completion.choices[0].message.content
    elif model_info['name'].startswith('Claude'):
        chat_completion = model_info['client'].messages.create(
            model=model_info['model'],
            messages=[m for m in messages if m['role'] != 'system'],
            system=sys_msg,
            max_tokens=1000
        )
        response = chat_completion.content[0].text
    else:
        raise ValueError(f"Unsupported model: {model_info['name']}")
    
    return response

def speak(text):
    global is_speaking
    try:
        url = "https://api.deepgram.com/v1/speak"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "text": text
        }
        params = {
            "model": "aura-luna-en"
        }
        
        response = requests.post(url, headers=headers, json=data, params=params)
        
        if response.status_code == 200:
            # Convert MP3 to WAV
            mp3_audio = AudioSegment.from_mp3(io.BytesIO(response.content))
            wav_audio = io.BytesIO()
            mp3_audio.export(wav_audio, format="wav")
            wav_audio.seek(0)
            
            # Play the generated audio
            p = pyaudio.PyAudio()
            wf = wave.open(wav_audio, 'rb')
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)

            is_speaking = True
            data = wf.readframes(CHUNK)
            while data and is_speaking:
                stream.write(data)
                data = wf.readframes(CHUNK)
        else:
            print(f"Error in speech synthesis: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error in speech synthesis: {e}")
    finally:
        is_speaking = False
        if 'stream' in locals():
            stream.stop_stream()
            stream.close()
        if 'p' in locals():
            p.terminate()
        if 'wf' in locals():
            wf.close()

def process_input(prompt, model_info):
    try:
        # Extract function calls without including them in the main prompt
        calls = function_call(prompt)
        
        # Remove function call triggers from the prompt
        clean_prompt = prompt
        for call in ['use your rag', 'take screenshot', 'capture webcam', 'extract clipboard']:
            clean_prompt = clean_prompt.replace(call, '').strip()
        
        visual_context = None
        rag_context = None
        clipboard_content = None
        rag_sources = None
        
        for call in calls:
            if call == 'take screenshot':
                print('Taking screenshot...')
                take_screenshot()
                latest_screenshot = get_latest_file('screenshot')
                visual_context = vision_prompt(prompt=clean_prompt, photo_path=latest_screenshot)
                print("\nImage Analysis:")
                print(process_image_data(visual_context))
            elif call == 'capture webcam':
                print('Capturing webcam...')
                web_cam_capture()
                latest_webcam = get_latest_file('webcam')
                visual_context = vision_prompt(prompt=clean_prompt, photo_path=latest_webcam)
                print("\nImage Analysis:")
                print(process_image_data(visual_context))
            elif call == 'extract clipboard':
                print('Copying clipboard text...')
                clipboard_content = get_clipboard_text()
            elif call == 'use rag':
                print('Using RAG data...')
                rag_prompt, rag_sources = query_rag(clean_prompt)
                rag_context = f"RAG Context:\n{rag_prompt}"
        
        # Add the current input to memory
        memory.add_memory(clean_prompt, importance=1)
        
        relevant_memories = memory.get_relevant_memories(clean_prompt)
        memory_context = "\n".join([m['content'] for m in relevant_memories])
        
        full_prompt = clean_prompt
        if memory_context:
            full_prompt = f"Relevant memories:\n{memory_context}\n\n{full_prompt}"
        if visual_context:
            full_prompt += f"\n\nIMAGE CONTEXT: {visual_context}"
        if rag_context:
            full_prompt += f"\n\n{rag_context}"
        if clipboard_content:
            full_prompt += f"\n\nCLIPBOARD CONTENT: {clipboard_content}"
        
        response = ai_prompt(prompt=full_prompt, img_context=visual_context, model_info=model_info)
        
        # Separate the main response from the sources
        main_response = response
        sources_text = ""
        if rag_sources:
            sources_text = f"\nSources: {rag_sources}"
        
        # Print the full response including sources
        print(f"\nAI: {main_response}{sources_text}")
        
        # Speak only the main response
        speak(main_response)
        
        # Add the full response (including sources) to memory
        memory.add_memory(f"{main_response}{sources_text}", importance=1)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        
def stop_speech():
    global is_speaking
    is_speaking = False

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will listen to the users instructions on extracting clipboard content, '
        'taking a screenshot, capturing the webcam, using RAG data, or calling no functions for a voice assistant to respond '
        'to the users prompt. The webcam is a virtual camera that the user connected to their iPhone 15 Pro. '
        'You can call multiple functions if the user request implies multiple actions. '
        'You will respond with a comma-separated list of selections from this list: ["extract clipboard", "take screenshot", "capture webcam", "use rag", "None"] \n'
        'Do not respond with anything but the most logical selection(s) from that list with no explanations. Format the '
        'function call names exactly as I listed, separated by commas if there are multiple.'
    )

    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]
    
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message.content

    return [call.strip() for call in response.split(',')]

def get_latest_file(directory):
    files = glob.glob(os.path.join(directory, '*'))
    return max(files, key=os.path.getctime)

def take_screenshot():
    folder_path = 'screenshot'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = os.path.join(folder_path, f'{timestamp}.jpg')

    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(filename, quality=95)
    print(f'Screenshot saved to {filename}')

def is_frame_black(frame):
    return np.sum(frame) == 0

def web_cam_capture():
    web_cam = cv2.VideoCapture(0)

    if not web_cam.isOpened():
        print('Error: Camera did not open successfully')
        return

    while True:
        ret, frame = web_cam.read()
        if not ret:
            print('Error: Failed to capture image')
            web_cam.release()
            return
        
        if not is_frame_black(frame):
            break

        print('Captured frame is black, trying again...')

    folder_path = 'webcam'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = os.path.join(folder_path, f'{timestamp}.png')

    cv2.imwrite(filename, frame)
    web_cam.release()
    print(f'Image saved to {filename}')

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str):
        return clipboard_content
    else:
            print('No clipboard text to copy')
            return None

def play_sound(frequency, duration):
    winsound.Beep(frequency, duration)

def record_audio():
    global is_recording, frames, recording_start_time
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    
    print("Recording started...")
    play_sound(1000, 100)  # Play a sound to indicate recording start
    frames = []
    recording_start_time = time.time()
    
    while is_recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()

def toggle_recording():
    global is_recording, recording_thread, frames
    
    with recording_lock:
        if not is_recording:
            is_recording = True
            recording_thread = threading.Thread(target=record_audio)
            recording_thread.start()
        else:
            is_recording = False
            if recording_thread:
                recording_thread.join()
            
            duration = time.time() - recording_start_time
            print(f"Recording stopped. Duration: {duration:.2f} seconds")
            play_sound(800, 100)  # Play a sound to indicate recording end
            
            if duration >= MIN_RECORDING_LENGTH:
                save_and_process_audio()
            else:
                print(f"Recording was too short ({duration:.2f} seconds). Please try again.")
                frames = []  # Clear the frames for the next attempt

def save_and_process_audio():
    global frames, selected_model
    if frames:
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        transcription = transcribe_audio()
        if transcription:
            print(f"Transcription: {transcription}")
            process_input(transcription, selected_model)
        else:
            print("No speech detected or transcription failed. Please try again.")
    else:
        print("No audio data recorded. Please try again.")

def transcribe_audio():
    global whisperx_model
    if os.path.exists(WAVE_OUTPUT_FILENAME):
        try:
            # Load audio
            audio = whisperx.load_audio(WAVE_OUTPUT_FILENAME)
            
            # Transcribe with WhisperX
            result = whisperx_model.transcribe(audio, batch_size=batch_size)
            
            # Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            
            # Clean up alignment model
            del model_a
            gc.collect()
            torch.cuda.empty_cache()
            
            # Extract transcription from segments
            transcription = " ".join([segment['text'] for segment in result["segments"]])
            
            return transcription
        except Exception as e:
            print(f"Error during transcription: {e}")
    return ""

def process_image_data(data):
    try:
        image_data = json.loads(data)
        scene = image_data.get('scene', '')
        caption = image_data.get('caption', '')
        return f"Scene: {scene}\nCaption: {caption}"
    except json.JSONDecodeError:
        return data  # If it's not JSON, return the original string
    
def is_correct_hotkey():
    return keyboard.is_pressed('left alt') and keyboard.is_pressed('right alt') and keyboard.is_pressed('f7')

def hotkey_handler(e):
    if is_correct_hotkey():
        toggle_recording()

def main():
    global is_speaking, selected_model, memory, whisperx_model

    # Initialize the memory system
    memory = AdvancedMemory()

    # Load WhisperX model
    
    whisperx_model = whisperx.load_model('base', device, compute_type=compute_type)

    # Register the hotkey handler
    keyboard.on_press(hotkey_handler)

    print("Press Left Alt + Right Alt + F7 to start/stop recording. Type 'exit' to quit.")
    
    # Select the model at the start
    selected_model = select_model()
    print(f"Using model: {selected_model['name']}")
    
    while True:
        try:
            prompt = input("USER: ").strip().lower()
            if prompt == 'exit':
                print("AI: Goodbye! It was nice chatting with you.")
                break
            elif prompt.startswith('remember:'):
                memory_content = prompt[9:].strip()
                memory.add_memory(memory_content, importance=2)
                print("AI: I've stored that information in my memory.")
            elif prompt == 'summarize memories':
                summary = memory.get_memory_summary()
                print(f"AI: Here's a summary of my recent memories:\n{summary}")
            elif prompt.startswith('update importance:'):
                content, importance = prompt[18:].rsplit(',', 1)
                if memory.update_memory_importance(content.strip(), int(importance)):
                    print("AI: Memory importance updated.")
                else:
                    print("AI: Memory not found.")
            elif prompt:
                process_input(prompt, selected_model)
        except KeyboardInterrupt:
            print("\nInterrupted. Stopping speech and waiting for new input.")
            stop_speech()  # Stop the speech when Ctrl+C is pressed
        except Exception as e:
            print(f"An error occurred: {e}")
        print()

if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up resources
        if whisperx_model:
            del whisperx_model
        gc.collect()
        torch.cuda.empty_cache()