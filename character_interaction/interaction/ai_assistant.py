
import os
from django.conf import settings
import pygame
import speech_recognition as sr
from gtts import gTTS
from google.cloud import texttospeech
from transformers import pipeline
import google.generativeai as genai
import concurrent.futures
import pronouncing  # Library for text-to-phoneme conversion
import re
import subprocess
import json
from pydub import AudioSegment  # Import pydub for audio conversion
import asyncio

# Set the path to the ffmpeg executable
ffmpeg_path = r"C:\Users\alimi\Downloads\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe"
os.environ["FFMPEG_BINARY"] = ffmpeg_path

class AI_Assistant:
    def __init__(self):
        genai.configure(api_key="AIzaSyAOc5EURc-Xp28JnItSImvT8q5sftnUun0")
        self.model = genai.GenerativeModel(
            model_name="models/gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 500,
                "response_mime_type": "text/plain",
            }
        )

        self.full_transcript = [
            {"role": "user", "parts": [
                """You are an AI avatar named Whatt, designed to be my best friend and a genius. Your responses should be emotionally expressive and suitable for lip-syncing with an avatar. For each response, follow these guidelines:

                1. Provide a natural, friendly response as Whatt.
                2. Express emotions appropriate to the context of the conversation.
                3. Keep responses concise, ideally under 50 words.
                4. After your main response, on a new line, include a phoneme dictionary with timing information. Use this format:

                PHONEMES: [
                { "start": 0.00, "end": 0.12, "value": "X" },
                { "start": 0.12, "end": 0.51, "value": "B" },
                ...
                ]

                The phoneme dictionary should cover the entire response, with realistic timing estimates. Use standard phoneme notation (e.g., IPA or ARPABET).

                Remember, you're a friendly AI, so keep your tone warm and engaging. Always stay in character as Whatt."""
            ]}
        ]
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()

        pygame.mixer.init()

        # self.emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True)
        
    def optimize_audio(self, input_file, output_file):
        audio = AudioSegment.from_file(input_file)
        # Convert to mono and reduce sample rate
        mono_audio = audio.set_channels(1)
        optimized_audio = mono_audio.set_frame_rate(16000)
        # Export the optimized audio
        optimized_audio.export(output_file, format="wav")
        
    async def run_rhubarb_async(self, command):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            # Use partial to correctly pass arguments to subprocess.run
            await loop.run_in_executor(pool, lambda: subprocess.run(command, shell=True, check=True))

    async def process_input(self, transcript):
        self.full_transcript.append({"role": "user", "parts": [transcript]})
        
        chat_session = self.model.start_chat(history=self.full_transcript)
        response = chat_session.send_message(transcript)
   
        ai_response = response.text.strip()
      
        split_text = re.split(r'PHONEMES:', ai_response)

        # The first part is the text before 'PHONEMES'
        text_part = split_text[0].strip()

        # The second part is the phonemes, so we add back 'PHONEMES:'
        phoneme_part = 'PHONEMES:' + split_text[1].strip()
        
        print(phoneme_part)
        self.full_transcript.append({"role": "model", "parts": [text_part]})

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # future_emotion = executor.submit(self.detect_emotion, ai_response)
            # emotion = future_emotion.result()

            audio_file = self.generate_audio(text_part)
            wav_file = audio_file.replace('.mp3', '.wav')

        # Convert mp3 to optimized wav
        self.optimize_audio(audio_file, wav_file)

        phonemes = self.extract_phonemes_from_text(ai_response)

        # Generate lip sync data using Rhubarb Lip Sync
        lip_sync_file_path = os.path.join(settings.MEDIA_ROOT, "response.json")
        rhubarb_exe_path = r"C:\Users\alimi\Downloads\Rhubarb-Lip-Sync-1.13.0-Windows\rhubarb.exe"
        rhubarb_command = f'"{rhubarb_exe_path}" -o "{lip_sync_file_path}" --threads 4 -f json "{wav_file}"'
        
        await self.run_rhubarb_async(rhubarb_command)

        # Generate the audio URL or path
        audio_url = f"{settings.MEDIA_URL}response.mp3"

        # Print the emotion and phonemes
        # print(f"Emotion: {emotion}")
        print("Phonemes:")
        for phoneme in phonemes:
            print(phoneme)

        return text_part, audio_url, phonemes, lip_sync_file_path

    def clean_text(self, text):
        # Remove unwanted characters (symbols and emojis)
        cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove symbols
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', cleaned_text)  # Remove non-ASCII characters (including emojis)
        return cleaned_text.strip()

    def detect_emotion(self, text):
        results = self.emotion_classifier(text)
        emotion = max(results[0], key=lambda x: x['score'])['label']
        return emotion

    def generate_audio(self, text):
        output_file = os.path.join(settings.MEDIA_ROOT, "response.mp3")
        try:
            self.generate_audio_google(text, output_file)
        except Exception as e:
            print(e)
            print("Error with Google TTS, using gTTS as fallback.")
            tts = gTTS(text=text, lang='en')
            tts.save(output_file)
        
        return output_file

    def generate_audio_google(self, text, output_file):
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-GB", name="en-GB-Studio-C"
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        with open(output_file, "wb") as out:
            out.write(response.audio_content)
        print(f"Audio content written to file {output_file}")

    def extract_phonemes_from_text(self, text):
        phonemes = []
        words = text.split()
        for word in words:
            phones = pronouncing.phones_for_word(word)
            if phones:
                phonemes.append({'word': word, 'phonemes': phones[0]})
            else:
                phonemes.append({'word': word, 'phonemes': []})
        return phonemes

    def play_audio(self, audio_file):
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()
        os.remove(audio_file)