import logging
import os

import speech_recognition as sr
from langchain_core.tools import tool
from pydub import AudioSegment

logger = logging.getLogger(__name__)


@tool
def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes speech from audio files to text using Google Speech Recognition API.

    Args:
        audio_path (str): Path to audio file

    Returns:
        str: Transcribed text
             On error: "Transcription error: <details>" or "Audio conversion error: <details>"

    Supported Formats:
    - Native: WAV
    - Auto-converted: MP3, MP4, FLAC, OGG, M4A

    Processing:
    - Non-WAV files are automatically converted to WAV using pydub
    - Conversion uses ffmpeg backend
    - Original file preserved; converted file saved as <name>.wav

    Limitations:
    - Requires internet connection (uses Google API)
    - Works best with clear speech in quiet environments

    Example:
        transcribe_audio("LLMFiles/clue.mp3")
    """

    logger.info(f"🎧 TRANSCRIBING: {audio_path}")
    try:
        # Convert to WAV if needed
        if not audio_path.lower().endswith(".wav"):
            try:
                audio = AudioSegment.from_file(audio_path)
                wav_path = os.path.splitext(audio_path)[0] + ".wav"
                audio.export(wav_path, format="wav")
                logger.info(f"🔄 Converted to WAV: {wav_path}")
                audio_path = wav_path
            except Exception as e:
                logger.error(f"❌ Conversion Failed: {e}")
                return f"Audio conversion error: {e}"

        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)

        logger.info(f"✅ Transcription: {text[:50]}...")
        return text

    except Exception as e:
        logger.error(f"💥 Transcription Failed: {e}")
        return f"Transcription error: {str(e)}"
