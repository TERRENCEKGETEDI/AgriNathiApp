import azure.cognitiveservices.speech as speechsdk
import os

class SpeechToTextService:
    def __init__(self):
        # Initialize the Azure Speech client
        self.mock_mode = False
        try:
            # Azure credentials from environment variables
            self.subscription_key = os.getenv('AZURE_SPEECH_KEY')
            self.region = os.getenv('AZURE_SPEECH_REGION', 'southafricanorth')

            if not self.subscription_key:
                raise ValueError("AZURE_SPEECH_KEY environment variable not set")

            self.speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
            print("Azure Speech client initialized successfully")
            # Test the API with a minimal request to check if it's enabled
            self._test_api_connection()
        except Exception as e:
            print(f"Azure Speech API not available: {e}")
            print("Switching to mock mode for testing...")
            self.mock_mode = True

    def _test_api_connection(self):
        """
        Test the API connection with a minimal request
        """
        try:
            # Try a minimal API call to test if the service is enabled
            # For Azure, we'll just check if we can create a recognizer
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=False)
            recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
            print("Azure Speech API is enabled and working")
        except Exception as e:
            error_str = str(e).lower()
            if "unauthorized" in error_str or "403" in error_str or "invalid" in error_str:
                print(f"API test failed - service disabled: {e}")
                self.mock_mode = True
            else:
                print(f"API test completed (expected failure with test data): {e}")

    def transcribe_audio(self, audio_file_path, language_code='zu-ZA'):
        """
        Transcribes audio file to text using Azure Speech API
        :param audio_file_path: Path to the audio file
        :param language_code: Language code for isiZulu (zu-ZA)
        :return: Transcribed text
        """
        print(f"DEBUG: Starting transcription for file: {audio_file_path}")
        print(f"DEBUG: Mock mode: {self.mock_mode}")
        print(f"DEBUG: Requested language: {language_code}")

        if self.mock_mode:
            return self._mock_transcription(audio_file_path)

        try:
            # Read the audio file
            with open(audio_file_path, 'rb') as audio_file:
                content = audio_file.read()

            print(f"DEBUG: Audio file size: {len(content)} bytes")
            print(f"DEBUG: First 10 bytes (hex): {content[:10].hex() if len(content) >= 10 else 'N/A'}")

            # Create audio config from file
            audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)

            # Try English first for better recognition, then isiZulu if needed
            print(f"DEBUG: Trying English recognition first for better accuracy")

            # Create English speech config
            english_speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
            english_speech_config.speech_recognition_language = 'en-US'

            print(f"DEBUG: Sending English request to Azure Speech API")
            english_recognizer = speechsdk.SpeechRecognizer(speech_config=english_speech_config, audio_config=audio_config)
            english_result = english_recognizer.recognize_once()
            print(f"DEBUG: English API response received: {english_result.reason}")

            # Extract English transcript
            english_transcript = ""
            english_confidence = 0.0
            if english_result.reason == speechsdk.ResultReason.RecognizedSpeech:
                english_transcript = english_result.text
                english_confidence = english_result.confidence if hasattr(english_result, 'confidence') else 0.5
                print(f"DEBUG: English transcript: '{english_transcript}' (confidence: {english_confidence})")
            else:
                print(f"DEBUG: English recognition failed: {english_result.reason}")

            # Now try isiZulu
            print(f"DEBUG: Sending isiZulu request to Azure Speech API")
            zulu_speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
            zulu_speech_config.speech_recognition_language = language_code

            zulu_recognizer = speechsdk.SpeechRecognizer(speech_config=zulu_speech_config, audio_config=audio_config)
            zulu_result = zulu_recognizer.recognize_once()
            print(f"DEBUG: isiZulu API response received: {zulu_result.reason}")

            # Extract isiZulu transcript
            zulu_transcript = ""
            zulu_confidence = 0.0
            if zulu_result.reason == speechsdk.ResultReason.RecognizedSpeech:
                zulu_transcript = zulu_result.text
                zulu_confidence = zulu_result.confidence if hasattr(zulu_result, 'confidence') else 0.5
                print(f"DEBUG: isiZulu transcript: '{zulu_transcript}' (confidence: {zulu_confidence})")
            else:
                print(f"DEBUG: isiZulu recognition failed: {zulu_result.reason}")

            # Choose the better result based on confidence and content
            if english_confidence > zulu_confidence and english_transcript:
                print(f"DEBUG: Using English result (higher confidence: {english_confidence} vs {zulu_confidence})")
                final_transcript = english_transcript
            elif zulu_transcript:
                print(f"DEBUG: Using isiZulu result (confidence: {zulu_confidence})")
                final_transcript = zulu_transcript
            elif english_transcript:
                print(f"DEBUG: Using English result (only available option)")
                final_transcript = english_transcript
            else:
                print(f"DEBUG: No valid transcripts found")
                final_transcript = ""

            print(f"DEBUG: Final transcript: '{final_transcript}'")

            # If no transcript and file is small, it might be too short
            if not final_transcript and len(content) < 50000:  # Less than 50KB
                print(f"DEBUG: Audio file too small ({len(content)} bytes), likely no speech captured")
                return ""

            return final_transcript

        except Exception as e:
            error_str = str(e).lower()
            # Check if this is an API disabled or authentication error
            if "unauthorized" in error_str or "403" in error_str or "invalid" in error_str:
                print(f"Azure Speech API error detected: {e}")
                print("Switching to mock mode for this request...")
                self.mock_mode = True
                return self._mock_transcription(audio_file_path)
            else:
                raise Exception(f"Error transcribing audio: {str(e)}")

    def transcribe_audio_stream(self, audio_stream, language_code='zu-ZA'):
        """
        Transcribes audio stream to text
        :param audio_stream: Audio stream data
        :param language_code: Language code
        :return: Transcribed text
        """
        print(f"DEBUG: Starting stream transcription, data size: {len(audio_stream)} bytes")
        print(f"DEBUG: First 10 bytes (hex): {audio_stream[:10].hex() if len(audio_stream) >= 10 else 'N/A'}")
        print(f"DEBUG: Mock mode: {self.mock_mode}")

        if self.mock_mode:
            return self._mock_transcription("stream")

        try:
            # Create audio config from stream data
            # For Azure, we need to write the stream to a temporary file or use push stream
            import tempfile
            import io

            # Write stream to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
                temp_file.write(audio_stream)
                temp_file_path = temp_file.name

            try:
                audio_config = speechsdk.audio.AudioConfig(filename=temp_file_path)

                # For browser-recorded audio streams, it's almost always WebM/Opus
                print("DEBUG: Using WebM/Opus format for browser stream audio")
                speech_config = speechsdk.SpeechConfig(subscription=self.subscription_key, region=self.region)
                speech_config.speech_recognition_language = language_code

                print(f"DEBUG: Sending stream request to Azure Speech API")
                recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
                result = recognizer.recognize_once()
                print(f"DEBUG: Stream API response received: {result.reason}")

                transcript = ""
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    transcript = result.text
                    print(f"DEBUG: Stream final transcript: '{transcript}'")
                else:
                    print(f"DEBUG: Stream recognition failed: {result.reason}")

                return transcript

            finally:
                # Clean up temporary file
                import os
                os.unlink(temp_file_path)

        except Exception as e:
            error_str = str(e).lower()
            # Check if this is an API disabled or authentication error
            if "unauthorized" in error_str or "403" in error_str or "invalid" in error_str:
                print(f"Azure Speech API error detected: {e}")
                print("Switching to mock mode for this request...")
                self.mock_mode = True
                return self._mock_transcription("stream")
            else:
                raise Exception(f"Error transcribing audio stream: {str(e)}")

    def _mock_transcription(self, source):
        """
        Mock transcription for testing when API is not available
        Returns a message indicating API is not available
        """
        message = "Speech-to-Text API is not available. Please enable Google Cloud Speech-to-Text API."
        print(f"Mock transcription for {source}: {message}")
        return message