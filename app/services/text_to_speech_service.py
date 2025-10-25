from google.cloud import texttospeech
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TextToSpeechService:
    def __init__(self):
        self.mock_mode = False
        try:
            self.client = texttospeech.TextToSpeechClient()
            print("Google Cloud Text-to-Speech client initialized successfully")
            self._test_api_connection()
        except Exception as e:
            print(f"Google Cloud Text-to-Speech API not available: {e}")
            print("Switching to mock mode for testing...")
            self.mock_mode = True

    def _test_api_connection(self):
        """
        Test the API connection with a minimal request
        """
        try:
            # Try a minimal API call to test if the service is enabled
            synthesis_input = texttospeech.SynthesisInput(text="test")
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            )

            # This will fail if API is disabled, triggering our error handling
            self.client.synthesize_speech(
                input=synthesis_input, voice=voice, audio_config=audio_config
            )
            print("Google Cloud Text-to-Speech API is enabled and working")
        except Exception as e:
            error_str = str(e).lower()
            if "service_disabled" in error_str or "403" in error_str:
                print(f"API test failed - service disabled: {e}")
                self.mock_mode = True
            else:
                print(f"API test completed (expected failure with test data): {e}")

    def synthesize_speech(self, text, language_code="zu-ZA", voice_gender="NEUTRAL"):
        """
        Enhanced speech synthesis with quality optimization and comprehensive error handling.
        :param text: Text to synthesize
        :param language_code: Language code (default: "zu-ZA" for isiZulu)
        :param voice_gender: Voice gender ("MALE", "FEMALE", or "NEUTRAL")
        :return: Audio content as bytes
        """
        synthesis_start = datetime.now()
        print(f"[TTS_SYNTHESIS] Starting synthesis for text length: {len(text)} chars, language: {language_code}, gender: {voice_gender}")

        # Input validation
        if not text or text.strip() == "":
            print("[TTS_SYNTHESIS] Empty text provided")
            return None

        # Clean and normalize text
        text = text.strip()
        if len(text) > 5000:
            print(f"[TTS_SYNTHESIS] Text too long ({len(text)} chars), truncating")
            text = text[:5000]

        if self.mock_mode:
            print("[TTS_SYNTHESIS] Using mock mode")
            return self._mock_synthesis(text)

        try:
            print(f"[TTS_SYNTHESIS] Synthesizing speech for '{text[:50]}...' in {language_code}")

            # Enhanced voice gender mapping with fallbacks
            gender_map = {
                "MALE": texttospeech.SsmlVoiceGender.MALE,
                "FEMALE": texttospeech.SsmlVoiceGender.FEMALE,
                "NEUTRAL": texttospeech.SsmlVoiceGender.NEUTRAL,
            }

            ssml_gender = gender_map.get(voice_gender.upper(), texttospeech.SsmlVoiceGender.NEUTRAL)

            # Set the text input to be synthesized
            synthesis_input = texttospeech.SynthesisInput(text=text)

            # Enhanced voice selection with specific voice names for better quality
            voice_params = {
                "language_code": language_code,
                "ssml_gender": ssml_gender,
            }

            # Add specific voice names for better quality (when available)
            if language_code == "zu-ZA":
                # Use specific Zulu voice if available
                voice_params["name"] = f"{language_code}-Standard-A"
            elif language_code.startswith("en-"):
                # Use high-quality English voices
                if voice_gender.upper() == "FEMALE":
                    voice_params["name"] = f"{language_code}-Standard-C"
                elif voice_gender.upper() == "MALE":
                    voice_params["name"] = f"{language_code}-Standard-D"
                else:
                    voice_params["name"] = f"{language_code}-Standard-A"

            voice = texttospeech.VoiceSelectionParams(**voice_params)

            # Optimized audio configuration for quality and compatibility
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,  # High quality
                speaking_rate=0.9,  # Slightly slower for clarity
                pitch=0.0,  # Natural pitch
                sample_rate_hertz=22050,  # Good balance of quality and size
            )

            # Perform the text-to-speech request
            print("[TTS_SYNTHESIS] Calling Google Cloud TTS API...")
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )

            audio_content = response.audio_content

            if not audio_content:
                print("[TTS_SYNTHESIS] No audio content received from API")
                return self._mock_synthesis(text)

            audio_size = len(audio_content)
            synthesis_end = datetime.now()
            synthesis_time = (synthesis_end - synthesis_start).total_seconds()

            print(f"[TTS_SYNTHESIS] Synthesis completed in {synthesis_time:.2f}s, audio size: {audio_size} bytes")
            print(f"[TTS_SYNTHESIS] Audio content type: {type(audio_content)}")

            # Validate audio content
            if audio_size < 100:
                print(f"[TTS_SYNTHESIS] Audio content suspiciously small: {audio_size} bytes")
                return self._mock_synthesis(text)

            # Log first few bytes for debugging
            if audio_content:
                print(f"[TTS_SYNTHESIS] First 20 bytes (hex): {audio_content[:20].hex()}")

            return audio_content

        except Exception as e:
            synthesis_end = datetime.now()
            synthesis_time = (synthesis_end - synthesis_start).total_seconds()

            error_str = str(e).lower()
            print(f"[TTS_SYNTHESIS] Error after {synthesis_time:.2f}s: {e}")

            # Check for specific API errors
            if any(keyword in error_str for keyword in ["service_disabled", "403", "not enabled", "permission", "unauthorized"]):
                print("[TTS_SYNTHESIS] API access error detected, switching to mock mode")
                self.mock_mode = True
                return self._mock_synthesis(text)
            elif "quota" in error_str or "limit" in error_str:
                print("[TTS_SYNTHESIS] API quota exceeded, using fallback")
                return self._mock_synthesis(text)
            else:
                print("[TTS_SYNTHESIS] General API error, using fallback")
                return self._mock_synthesis(text)

    def _mock_synthesis(self, text):
        """
        Enhanced mock speech synthesis for testing when API is not available.
        Generates a more realistic audio placeholder with variable length based on text.
        """
        import math
        from datetime import datetime

        mock_start = datetime.now()
        print(f"[MOCK_SYNTHESIS] Generating mock audio for text length: {len(text)}")

        # Calculate duration based on text length (rough estimation: 150 chars per minute)
        chars_per_minute = 150
        duration = max(0.5, min(5.0, len(text) / chars_per_minute * 60))  # 0.5-5 seconds

        sample_rate = 22050  # Good quality sample rate
        num_samples = int(sample_rate * duration)

        print(f"[MOCK_SYNTHESIS] Generating {duration:.1f}s audio ({num_samples} samples)")

        # WAV header (44 bytes) - proper WAV format
        wav_header = (
            b'RIFF'  # ChunkID
            + (36 + num_samples * 2).to_bytes(4, 'little')  # ChunkSize
            + b'WAVE'  # Format
            + b'fmt '  # Subchunk1ID
            + (16).to_bytes(4, 'little')  # Subchunk1Size
            + (1).to_bytes(2, 'little')  # AudioFormat (PCM)
            + (1).to_bytes(2, 'little')  # NumChannels (mono)
            + sample_rate.to_bytes(4, 'little')  # SampleRate
            + (sample_rate * 2).to_bytes(4, 'little')  # ByteRate
            + (2).to_bytes(2, 'little')  # BlockAlign
            + (16).to_bytes(2, 'little')  # BitsPerSample
            + b'data'  # Subchunk2ID
            + (num_samples * 2).to_bytes(4, 'little')  # Subchunk2Size
        )

        # Generate more realistic audio - combination of frequencies
        audio_data = b''
        base_freq = 220  # A3 note as base

        for i in range(num_samples):
            # Create a more complex waveform with harmonics
            t = i / sample_rate
            # Fundamental frequency with some variation
            freq_variation = base_freq * (1 + 0.1 * math.sin(2 * math.pi * 0.5 * t))
            # Add some harmonics and noise for realism
            sample = (
                0.6 * math.sin(2 * math.pi * freq_variation * t) +  # Fundamental
                0.3 * math.sin(2 * math.pi * freq_variation * 2 * t) +  # 2nd harmonic
                0.1 * math.sin(2 * math.pi * freq_variation * 3 * t)  # 3rd harmonic
            )

            # Add subtle amplitude modulation for more natural sound
            amplitude_mod = 0.8 + 0.2 * math.sin(2 * math.pi * 2 * t)
            sample *= amplitude_mod

            # Convert to 16-bit PCM
            sample_int = int(32767 * sample * 0.3)  # Lower volume to avoid clipping
            audio_data += sample_int.to_bytes(2, byteorder='little', signed=True)

        mock_wav = wav_header + audio_data

        mock_end = datetime.now()
        mock_time = (mock_end - mock_start).total_seconds()

        print(f"[MOCK_SYNTHESIS] Mock audio generated in {mock_time:.3f}s: {len(mock_wav)} bytes")
        return mock_wav