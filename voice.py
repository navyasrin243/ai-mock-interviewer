import os
from dotenv import load_dotenv

load_dotenv()


def transcribe_audio(audio_bytes: bytes) -> str:
    """Audio bytes -> text using Groq Whisper."""
    if not audio_bytes:
        return ""

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY missing")
        return ""

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        result = client.audio.transcriptions.create(
            file=("answer.wav", audio_bytes),
            model="whisper-large-v3-turbo",
            language="en",
        )
        return (getattr(result, "text", result) or "").strip()

    except Exception as e:
        print(f"Transcription failed: {e}")
        return ""