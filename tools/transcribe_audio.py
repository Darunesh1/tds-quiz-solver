import os

import speech_recognition as sr
from langchain_core.tools import tool
from pydub import AudioSegment


@tool
def transcribe_audio(audio_path: str) -> str:
    """Transcribe any audio file (MP3/WAV) to text using speech recognition."""
    try:
        # Convert to WAV
        if not audio_path.endswith(".wav"):
            audio = AudioSegment.from_file(audio_path)
            wav_path = audio_path.rsplit(".", 1)[0] + ".wav"
            audio.export(wav_path, format="wav")
            audio_path = wav_path

        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
        return text
    except Exception as e:
        return f"Transcription error: {str(e)}"
