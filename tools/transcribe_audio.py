"""Audio transcription tool using Google Speech Recognition."""

import logging
import os

import speech_recognition as sr
from langchain_core.tools import tool
from pydub import AudioSegment

logger = logging.getLogger(__name__)


@tool
def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribes speech from audio files to text using Google Speech Recognition.

    **When to Use:**
    - Page contains audio files (.mp3, .wav, .ogg)
    - Instructions mention "listen to" or "audio clue"
    - Visual content seems incomplete (audio may have missing information)
    - Challenge involves spoken passwords, codes, or instructions

    **How CTF Challenges Use Audio:**
    1. **Spoken Questions**: Question read aloud instead of written
    2. **Hidden Clues**: Passwords or codes spoken in audio
    3. **Multi-Step Puzzles**: Audio provides one part of a multi-part answer
    4. **Obfuscation**: Harder to scrape than text, requires transcription

    **Workflow:**
    ```
    # Step 1: Find audio file
    page = get_rendered_html("https://quiz.com/level")
    audio_url = page['assets']['audio'][0]

    # Step 2: Download audio
    file_info = download_file(audio_url)

    # Step 3: Transcribe
    result = transcribe_audio(file_info['filepath'])
    print(result['text'])
    ```

    **Supported Formats:**
    - **Native**: WAV (fastest, no conversion)
    - **Auto-Converted**: MP3, MP4, FLAC, OGG, M4A, AAC
    - Conversion uses ffmpeg (must be installed)

    **Transcription Quality:**
    - **Best**: Clear speech, quiet background, standard accent
    - **Good**: Normal conversation, minimal noise
    - **Poor**: Music background, multiple speakers, heavy accent
    - **Fails**: Pure music, very noisy audio, non-English (depends on Google API)

    Args:
        audio_path: Path to audio file (from download_file)
                   Can be relative path or just filename

    Returns:
        dict: {
            'text': str,  # Transcribed speech
            'word_count': int,  # Number of words detected
            'format': str,  # Original audio format
            'duration': float,  # Audio length in seconds (if available)
            'converted': bool  # True if format conversion was needed
        }
        On error: dict with 'error' field and 'suggestion'

    **Important Notes:**
    - Requires internet connection (uses Google's API)
    - No API key needed (uses free tier)
    - Long audio files (>60s) may need chunking
    - Non-WAV files converted automatically (creates .wav file)

    **Troubleshooting:**
    - "Audio conversion error": ffmpeg not installed
    - "Could not understand audio": Check if file contains speech
    - "Request error": Network issue, retry after a moment
    - Empty result: Audio may be too quiet or contain no speech
    """
    logger.info(f"🎧 TRANSCRIBING: {audio_path}")

    try:
        # Handle path variations
        if not os.path.exists(audio_path) and not audio_path.startswith("LLMFiles"):
            alt_path = os.path.join("LLMFiles", audio_path)
            if os.path.exists(alt_path):
                audio_path = alt_path

        # Verify file exists
        if not os.path.exists(audio_path):
            error_msg = f"Audio file not found: {audio_path}"
            logger.error(f"💥 {error_msg}")
            llm_files = os.listdir("LLMFiles") if os.path.exists("LLMFiles") else []
            return {
                "error": error_msg,
                "suggestion": f"Available files: {llm_files}",
                "filepath": audio_path,
            }

        original_format = os.path.splitext(audio_path).lower()[3]
        converted = False
        duration = None

        # Convert to WAV if needed
        if not audio_path.lower().endswith(".wav"):
            try:
                logger.info(f"   Converting {original_format} to WAV...")
                audio = AudioSegment.from_file(audio_path)
                wav_path = os.path.splitext(audio_path) + ".wav"
                audio.export(wav_path, format="wav")
                duration = len(audio) / 1000.0  # Convert ms to seconds
                audio_path = wav_path
                converted = True
                logger.info(f"   ✓ Converted to: {wav_path} ({duration:.1f}s)")
            except Exception as e:
                logger.error(f"❌ Conversion Failed: {e}")
                return {
                    "error": f"Audio conversion failed: {str(e)}",
                    "suggestion": "Ensure ffmpeg is installed: sudo apt-get install ffmpeg",
                    "filepath": audio_path,
                }

        # Transcribe using Google Speech Recognition
        recognizer = sr.Recognizer()

        with sr.AudioFile(audio_path) as source:
            logger.info("   Analyzing audio...")
            audio_data = recognizer.record(source)

        logger.info("   Sending to Google Speech API...")
        text = recognizer.recognize_google(audio_data)

        word_count = len(text.split())
        logger.info(f"✅ Transcription Success: {word_count} words")
        logger.info(f"   Preview: {text[:100]}...")

        return {
            "text": text,
            "word_count": word_count,
            "format": original_format,
            "duration": duration,
            "converted": converted,
            "filepath": audio_path,
            "success": True,
        }

    except sr.UnknownValueError:
        error_msg = "Could not understand audio (no clear speech detected)"
        logger.warning(f"⚠️ {error_msg}")
        return {
            "error": error_msg,
            "suggestion": "Audio may contain no speech, be too noisy, or be in unsupported language",
            "text": "",  # Return empty string rather than error
            "success": False,
        }

    except sr.RequestError as e:
        error_msg = f"Google API request failed: {str(e)}"
        logger.error(f"💥 {error_msg}")
        return {
            "error": error_msg,
            "suggestion": "Check internet connection and retry",
            "filepath": audio_path,
        }

    except Exception as e:
        logger.error(f"💥 Transcription Failed: {e}")
        return {
            "error": str(e),
            "suggestion": "Verify audio file format and integrity",
            "filepath": audio_path,
        }
