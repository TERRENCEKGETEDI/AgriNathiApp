from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import os
import base64
import sys
import json
import tempfile
import logging
from datetime import datetime


def create_app():
    app = Flask(__name__)

    # Enhanced logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Only use stream handler for Render
        ]
    )
    app.logger = logging.getLogger('agri_nathi')
    app.logger.info("Starting AgriNathi application")

    # Configuration
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    app.config['UPLOAD_FOLDER'] = 'data/audio_recordings'

    # Session configuration
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_PERMANENT'] = False
    app.config['SESSION_USE_SIGNER'] = True

    # Initialize session
    # Note: In production, use a proper session store like Redis

    # Import the actual voice recognition service
    try:
        # Set Google Cloud credentials environment variable
        credentials_path = os.path.join(os.path.dirname(__file__), '..', 'google-credentials.json')
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        print(f"Google Cloud credentials path: {credentials_path}")

        # Import services from the local app/services directory
        import sys
        sys.path.append('.')
        from app.services.speech_service import SpeechToTextService
        from app.services.translation_service import TranslationService
        from app.services.text_to_speech_service import TextToSpeechService
        from app.services.agriculture_ml_service import AgricultureMLService

        speech_service = SpeechToTextService()
        translation_service = TranslationService()
        tts_service = TextToSpeechService()
        ml_service = AgricultureMLService()
        print("Google Cloud services initialized successfully!")
        use_mock = False

    except Exception as e:
        print(f"Could not import services: {e}. Using mock mode.")
        speech_service = None
        translation_service = None
        tts_service = None
        ml_service = None
        use_mock = True

    class VoiceRecognition:
        def __init__(self):
            self.use_mock = use_mock
            self.tts_service = tts_service if not use_mock else None
            self.ml_service = ml_service if not use_mock else None

        def process_audio(self, audio_base64):
            """
            Enhanced voice processing pipeline with senior developer expertise.
            Process: Audio → Transcription → Translation → Response Generation → Audio Synthesis
            """
            if self.use_mock:
                return self._mock_process_audio()

            processing_start = datetime.now()
            print(f"[VOICE_PROCESSING] Starting audio processing at {processing_start}")

            try:
                # Step 1: Decode and validate audio data
                print("[VOICE_PROCESSING] Step 1: Decoding audio data")
                audio_data = base64.b64decode(audio_base64)
                if len(audio_data) < 100:  # Basic validation
                    raise ValueError("Audio data too small, likely corrupted")

                # Step 2: Save temporary audio file with better error handling
                print("[VOICE_PROCESSING] Step 2: Saving temporary audio file")
                temp_file_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                        temp_file.write(audio_data)
                        temp_file_path = temp_file.name
                        print(f"[VOICE_PROCESSING] Temporary file created: {temp_file_path}")

                    # Step 3: Transcribe audio with enhanced error handling
                    print("[VOICE_PROCESSING] Step 3: Transcribing audio")
                    transcript = speech_service.transcribe_audio(temp_file_path, language_code='zu-ZA')

                    if not transcript or transcript.strip() == "":
                        print("[VOICE_PROCESSING] No speech detected, using fallback message")
                        transcript = "Speech not detected. Please try speaking louder and clearer."
                        translation = "Speech not detected. Please try speaking louder and clearer."
                        advice = "Please speak clearly and ensure your microphone is working properly."
                        advice_zulu = "Sicela ukhulume ngokusobala futhi uqinisekise ukuthi imakrofoni yakho iyasebenza kahle."
                    else:
                        print(f"[VOICE_PROCESSING] Transcription successful: '{transcript}'")

                        # Step 4: Translate to English
                        print("[VOICE_PROCESSING] Step 4: Translating to English")
                        translation = translation_service.translate_text(transcript, source_lang="zu", target_lang="en")
                        print(f"[VOICE_PROCESSING] Translation: '{translation}'")

                        # Step 5: Generate intelligent agricultural response
                        print("[VOICE_PROCESSING] Step 5: Generating agricultural advice")
                        advice = self._generate_ml_advice(transcript.lower())
                        print(f"[VOICE_PROCESSING] Generated advice: '{advice}'")

                        # Step 6: Translate advice back to isiZulu
                        print("[VOICE_PROCESSING] Step 6: Translating advice to isiZulu")
                        advice_zulu = translation_service.translate_text(advice, source_lang="en", target_lang="zu")
                        print(f"[VOICE_PROCESSING] Zulu advice: '{advice_zulu}'")

                    # Step 7: Generate audio responses (both languages)
                    print("[VOICE_PROCESSING] Step 7: Synthesizing speech")
                    audio_data = None
                    english_audio_data = None

                    if self.tts_service:
                        try:
                            # Primary response in isiZulu
                            audio_data = self.tts_service.synthesize_speech(advice_zulu, language_code="zu-ZA")
                            print(f"[VOICE_PROCESSING] Zulu audio synthesized: {len(audio_data) if audio_data else 0} bytes")

                            # Secondary response in English for reference
                            english_audio_data = self.tts_service.synthesize_speech(advice, language_code="en-US")
                            print(f"[VOICE_PROCESSING] English audio synthesized: {len(english_audio_data) if english_audio_data else 0} bytes")
                        except Exception as tts_error:
                            print(f"[VOICE_PROCESSING] TTS synthesis failed: {tts_error}")
                            # Continue without audio rather than failing completely

                    processing_end = datetime.now()
                    processing_time = (processing_end - processing_start).total_seconds()
                    print(f"[VOICE_PROCESSING] Processing completed in {processing_time:.2f} seconds")

                    return {
                        'success': True,
                        'transcript': transcript,
                        'translation': translation,
                        'advice': advice,
                        'advice_zulu': advice_zulu,
                        'audio': base64.b64encode(audio_data).decode('utf-8') if audio_data else None,
                        'english_audio': base64.b64encode(english_audio_data).decode('utf-8') if english_audio_data else None,
                        'processing_time': processing_time,
                        'timestamp': processing_end.isoformat()
                    }

                finally:
                    # Step 8: Clean up temporary file
                    if temp_file_path and os.path.exists(temp_file_path):
                        try:
                            os.unlink(temp_file_path)
                            print(f"[VOICE_PROCESSING] Temporary file cleaned up: {temp_file_path}")
                        except Exception as cleanup_error:
                            print(f"[VOICE_PROCESSING] Warning: Failed to clean up temp file: {cleanup_error}")

            except Exception as e:
                processing_end = datetime.now()
                processing_time = (processing_end - processing_start).total_seconds()
                error_msg = f'Failed to process audio: {str(e)}'
                print(f"[VOICE_PROCESSING] Error after {processing_time:.2f} seconds: {error_msg}")

                return {
                    'success': False,
                    'error': error_msg,
                    'transcript': '',
                    'translation': '',
                    'advice': 'Please try recording again. If the problem persists, contact support.',
                    'advice_zulu': 'Sicela uzame ukurekhoda futhi. Uma inkinga iqhubeka, xhumana nosizo.',
                    'audio': None,
                    'processing_time': processing_time,
                    'timestamp': processing_end.isoformat()
                }

        def _mock_process_audio(self):
            """
            Enhanced mock processing with intelligent response generation.
            Uses ML service to generate realistic responses even in mock mode.
            """
            mock_transcript = 'Sawubona, ngicela usizo ngezitshalo zami'
            mock_translation = 'Hello, I need help with my plants'

            # Generate intelligent advice using ML service even in mock mode
            try:
                if self.ml_service:
                    mock_advice = self.ml_service.generate_advice_text("plant care and diseases")
                    if mock_advice and len(mock_advice.strip()) > 20:
                        # Translate advice to Zulu for consistency
                        mock_advice_zulu = translation_service.translate_text(mock_advice, source_lang="en", target_lang="zu")
                    else:
                        # Fallback to enhanced keyword-based advice
                        mock_advice = self._generate_keyword_advice("izitshalo zifo")
                        mock_advice_zulu = translation_service.translate_text(mock_advice, source_lang="en", target_lang="zu")
                else:
                    # Use keyword-based advice
                    mock_advice = self._generate_keyword_advice("izitshalo zifo")
                    mock_advice_zulu = translation_service.translate_text(mock_advice, source_lang="en", target_lang="zu")
            except Exception as e:
                print(f"[MOCK_PROCESSING] Error generating advice: {e}")
                # Ultimate fallback
                mock_advice = 'For plant diseases, ensure proper watering and use organic pesticides.'
                mock_advice_zulu = 'Ngezifo zezitshalo, qinisekisa ukunisela okufanele futhi use izibulala-zinambuzane zemvelo.'

            print(f"[MOCK_PROCESSING] Generated advice: '{mock_advice[:50]}...'")

            return {
                'success': True,
                'transcript': mock_transcript,
                'translation': mock_translation,
                'advice': mock_advice,
                'advice_zulu': mock_advice_zulu,
                'audio': None,
                'english_audio': None,
                'mock_mode': True
            }

        def _generate_ml_advice(self, zulu_text):
            """
            Enhanced agricultural advice generation with intelligent processing.
            Uses ML service when available, falls back to sophisticated keyword matching.
            """
            print(f"[ADVICE_GENERATION] Processing query: '{zulu_text}'")

            if self.ml_service:
                try:
                    # First translate to English for better ML matching
                    english_text = translation_service.translate_text(zulu_text, source_lang="zu", target_lang="en")
                    print(f"[ADVICE_GENERATION] Translated query for ML: '{english_text}'")

                    # Step 1: Find specific solution using ML analysis
                    print("[ADVICE_GENERATION] Finding specific solution...")
                    ml_solution = self.ml_service.find_solution(english_text)
                    if ml_solution:
                        print(f"[ADVICE_GENERATION] Found ML solution: {ml_solution['solution'].get('name', 'Unknown')}")
                        # Use the specific solution found by ML analysis
                        solution = ml_solution['solution']
                        category = ml_solution['category']

                        # Generate advice based on the specific solution found
                        if category == 'diseases':
                            advice = f"For {solution.get('name', 'plant disease')}: {', '.join(solution.get('solutions', ['Consult agricultural expert']))}. Symptoms include: {', '.join(solution.get('symptoms', ['various signs']))}."
                        elif category == 'pests':
                            advice = f"For {solution.get('name', 'pest problem')}: {', '.join(solution.get('solutions', ['Use appropriate pest control']))}. Look for: {', '.join(solution.get('symptoms', ['pest signs']))}."
                        elif category == 'fertilizers':
                            advice = f"For fertilization: {solution.get('name', 'fertilizer')} - {', '.join(solution.get('usage', ['Apply as directed']))}. Benefits: {', '.join(solution.get('benefits', ['improved plant health']))}."
                        elif category == 'watering':
                            advice = f"Watering guidance: {solution.get('advice', 'Water appropriately')}. Tips: {', '.join(solution.get('tips', ['monitor soil moisture']))}."
                        elif category == 'planting':
                            advice = f"Planting advice for {solution.get('crop', 'crops')}: Plant in {solution.get('season', 'appropriate season')}. Spacing: {solution.get('spacing', 'follow guidelines')}. {solution.get('tips', ['Follow best practices'])}."
                        else:
                            # Fallback to general advice generation
                            advice = self.ml_service.generate_advice_text(english_text)

                        print(f"[ADVICE_GENERATION] Solution-based advice: '{advice}'")
                        return advice
                    else:
                        # No specific solution found, generate general advice
                        print("[ADVICE_GENERATION] No specific solution found, generating general advice...")
                        advice = self.ml_service.generate_advice_text(english_text)
                        print(f"[ADVICE_GENERATION] General ML advice: '{advice}'")
                        return advice

                except Exception as ml_error:
                    print(f"[ADVICE_GENERATION] ML service failed: {ml_error}, using fallback")
                    return self._generate_keyword_advice(zulu_text)

            # Enhanced fallback to keyword-based matching
            return self._generate_keyword_advice(zulu_text)

        def _generate_keyword_advice(self, zulu_text):
            """
            Enhanced keyword-based agricultural advice with intelligent matching and comprehensive responses.
            """
            print(f"[KEYWORD_ADVICE] Analyzing text: '{zulu_text}'")

            # Enhanced keyword mapping with multiple variations and priorities
            advice_map = {
                # Primary agricultural concerns
                'izitshalo': {
                    'priority': 1,
                    'advice': 'For plant care: Ensure proper watering, use organic fertilizers, and monitor for pests regularly. Regular soil testing helps maintain optimal nutrient levels.',
                    'zulu': 'Ngokunakekela izitshalo: Qinisekisa ukunisela okufanele, sebenzisa umanyolo wemvelo, futhi ubheke izinambuzane njalo. Ukuhlola inhlabathi njalo kusiza ukugcina amazinga afanele ezakhi.'
                },
                'zifo': {
                    'priority': 1,
                    'advice': 'For plant diseases: Remove affected leaves immediately, improve air circulation, and use copper-based fungicides. Early detection prevents spread to healthy plants.',
                    'zulu': 'Ngezifo zezitshalo: Susa amaqabunga athintekile ngokushesha, thuthukisa ukuhamba komoya, futhi use ama-fungicide asuselwe ku-copper. Ukuthola kusenesikhathi kuvimbela ukusabalala ezitshalweni ezinempilo.'
                },
                'nambuzane': {
                    'priority': 1,
                    'advice': 'For pest control: Use neem oil spray, introduce beneficial insects, and practice proper crop rotation. Integrated pest management reduces chemical dependency.',
                    'zulu': 'Ngokulawula izinambuzane: Sebenzisa isifutho se-neem oil, ngenisa izinambuzane ezisizayo, futhi wenze ukushintshana kwezitshalo. Ukuphathwa okudidiyelwe kwezinambuzane kunciphisa ukuncika kumakhemikhali.'
                },

                # Watering and irrigation
                'nisela': {
                    'priority': 2,
                    'advice': 'Watering advice: Water deeply but infrequently, early morning is best, avoid wetting leaves to prevent fungal diseases. Use drip irrigation for water efficiency.',
                    'zulu': 'Iseluleko sokunisela: Nisele ngokujulile kodwa kungavamile, ekuseni kuyinhle kakhulu, gwema ukumanzisa amaqabunga ukuvimbela izifo zokungcola. Sebenzisa i-drip irrigation ukuze ulondoloze amanzi.'
                },

                # Fertilizers and nutrients
                'umanyolo': {
                    'priority': 2,
                    'advice': 'Fertilizer guidance: Use balanced NPK fertilizer, apply during growing season, test soil pH first. Organic compost improves soil structure and microbial activity.',
                    'zulu': 'Ukuholwa ngomanyolo: Sebenzisa umanyolo obhalansiwe we-NPK, faka ngesikhathi sokukhula, hlola i-pH yenhlabathi kuqala. Umquba wemvelo uthuthukisa ukwakheka kwenhlabathi kanye nomsebenzi we-microbial.'
                },

                # Planting and seeds
                'imbewu': {
                    'priority': 2,
                    'advice': 'Seed planting: Plant during correct season, ensure proper spacing, keep soil moist until germination. Use certified seeds for better yields.',
                    'zulu': 'Ukubeka imbewu: Tshala ngesikhathi esifanele, qinisekisa isikhala esifanele, gcina inhlabathi imanzi kuze kube yilapho imbewu imila. Sebenzisa imbewu eqinisekisiwe ukuze uthole isivuno esingcono.'
                },

                # Weather and climate
                'isimo sezulu': {
                    'priority': 2,
                    'advice': 'Weather considerations: Monitor forecasts, protect crops from frost, prepare drainage for heavy rain. Climate-smart agriculture adapts to changing weather patterns.',
                    'zulu': 'Ukucabangela isimo sezulu: Buka izibikezelo, vikela izitshalo eqhweni, lungiselela ukukhipha amanzi emvuleni enkulu. Ezolimo ezihlakaniphile ngokwemvelo zivumelana namaphethini esimo sezulu ashintshayo.'
                },

                # Specific crops
                'khuni': {
                    'priority': 3,
                    'advice': 'Maize care: Plant in well-drained soil, fertilize regularly, watch for corn borer and rust diseases. Harvest at 20-25% moisture content for optimal storage.',
                    'zulu': 'Ukunakekela ummbila: Tshala enhlabathini ekhipha amanzi kahle, faka umanyolo njalo, bheka i-corn borer nezifo ze-rust. Vuna uma kunomswakama we-20-25% wokugcina okungcono.'
                },

                # Weed management
                'utshani': {
                    'priority': 3,
                    'advice': 'Weed control: Use mulching, hand weeding, or organic herbicides. Prevent weed competition for nutrients. Mechanical cultivation disrupts weed growth cycles.',
                    'zulu': 'Ukulawula utshani: Sebenzisa i-mulching, ukulima ngesandla, noma ama-herbicide emvelo. Vimbela ukuncintisana kotshani ngamaminerali. Ukulima ngemishini kuphazamisa imijikelezo yokukhula kotshani.'
                },

                # Soil management
                'umhlaba': {
                    'priority': 3,
                    'advice': 'Soil management: Test soil pH regularly, add organic matter, practice conservation tillage. Healthy soil is the foundation of successful farming.',
                    'zulu': 'Ukuphathwa kwenhlabathi: Hlola i-pH yenhlabathi njalo, engeza izinto eziphilayo, wenze i-conservation tillage. Inhlabathi enempilo iyisisekelo sezolimo eziphumelelayo.'
                },

                # Harvesting
                'isivuno': {
                    'priority': 3,
                    'advice': 'Harvesting: Harvest at correct maturity, use proper tools, store in cool dry place. Post-harvest handling affects final product quality.',
                    'zulu': 'Ukuvuna: Vuna lapho kuvuthiwe kahle, sebenzisa amathuluzi afanele, gcina endaweni epholile futhi eyomile. Ukuphathwa kwangemva kokuvuna kuthinta ikhwalithi yomkhiqizo wokugcina.'
                },

                # Livestock
                'izilwane': {
                    'priority': 3,
                    'advice': 'Livestock care: Provide clean water, balanced feed, regular health checks, proper housing. Animal health directly impacts farm productivity.',
                    'zulu': 'Ukunakekela izilwane: Nikeza amanzi ahlanzekile, ukudla okubhalansiwe, ukuhlola impilo njalo, nendawo yokuhlala efanele. Impilo yezilwane ithinta ngokuqondile ukukhiqiza kwepulazi.'
                }
            }

            # Multi-keyword analysis for better matching
            matched_keywords = []
            for keyword, data in advice_map.items():
                if keyword in zulu_text.lower():
                    matched_keywords.append((keyword, data))

            if matched_keywords:
                # Sort by priority and return the highest priority match
                matched_keywords.sort(key=lambda x: x[1]['priority'])
                best_match = matched_keywords[0][1]

                print(f"[KEYWORD_ADVICE] Matched keyword: {matched_keywords[0][0]}")
                return best_match['advice']

            # Use ML service for more intelligent responses when no keywords match
            print("[KEYWORD_ADVICE] No keywords matched, trying ML service")
            if self.ml_service:
                try:
                    # Translate to English for better ML matching
                    english_text = translation_service.translate_text(zulu_text, source_lang="zu", target_lang="en")
                    print(f"[KEYWORD_ADVICE] Translated for ML: '{english_text}'")

                    # Generate dynamic advice using ML service
                    ml_advice = self.ml_service.generate_advice_text(english_text)
                    if ml_advice and len(ml_advice.strip()) > 20:
                        print(f"[KEYWORD_ADVICE] Using ML-generated advice")
                        return ml_advice
                except Exception as ml_error:
                    print(f"[KEYWORD_ADVICE] ML service failed: {ml_error}")

            # Enhanced default advice with more specific guidance
            default_advices = [
                'General farming advice: Practice sustainable agriculture, monitor your crops regularly, maintain soil health, and seek local extension services for specific guidance. Sustainable farming practices ensure long-term productivity and environmental health.',
                'Farming best practices: Ensure proper crop rotation, use organic fertilizers when possible, maintain adequate soil moisture, and regularly inspect plants for pests and diseases. Early intervention prevents major crop losses.',
                'Agricultural recommendations: Test your soil pH annually, apply balanced fertilizers based on soil test results, practice integrated pest management, and keep detailed records of your farming activities for better planning.',
                'Crop management guidance: Monitor weather patterns closely, protect young plants from extreme conditions, use certified seeds, and maintain proper plant spacing for optimal growth and disease prevention.'
            ]

            # Select advice based on input characteristics
            if len(zulu_text.split()) > 3:  # Longer queries get more specific advice
                default_advice = default_advices[1]
            elif 'help' in zulu_text.lower() or 'usizo' in zulu_text.lower():
                default_advice = default_advices[2]
            else:
                default_advice = default_advices[0]

            print("[KEYWORD_ADVICE] Using enhanced default advice")
            return default_advice

        def get_advice_audio(self, advice_text):
            """Generate audio for agricultural advice in isiZulu"""
            # Generate dynamic advice based on input text
            if self.ml_service:
                advice_text = self.ml_service.generate_advice_text(advice_text)

            advice_zulu = translation_service.translate_text(advice_text, source_lang="en", target_lang="zu")

            audio_data = None
            if self.tts_service:
                audio_data = self.tts_service.synthesize_speech(advice_zulu, language_code="zu-ZA")

            return {
                'advice': advice_text,
                'advice_zulu': advice_zulu,
                'audio': base64.b64encode(audio_data).decode('utf-8') if audio_data else None
            }

    voice_recognition = VoiceRecognition()

    # Simple user storage (in production, use a database)
    users_db = {}

    # Helper functions
    def login_required(f):
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        decorated_function.__name__ = f.__name__
        return decorated_function

    def admin_required(f):
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session:
                return redirect(url_for('login'))
            user_email = session.get('user_email')
            user = users_db.get(user_email)
            if not user or user.get('role') != 'admin':
                flash('Access denied. Admin privileges required.', 'error')
                return redirect(url_for('index'))
            return f(*args, **kwargs)
        decorated_function.__name__ = f.__name__
        return decorated_function

    def load_users():
        """Load users from file (simple persistence)"""
        try:
            with open('data/users.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_users(users):
        """Save users to file"""
        os.makedirs('data', exist_ok=True)
        with open('data/users.json', 'w') as f:
            json.dump(users, f, indent=2)

    # Load users on startup
    users_db = load_users()

    # Authentication Routes
    @app.route('/login')
    def login():
        if 'user_id' in session:
            return redirect(url_for('index'))
        return render_template('login.html')

    @app.route('/login', methods=['POST'])
    def login_post():
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'}), 400

        # Check if user exists and password matches
        user = users_db.get(email)
        if user and user['password'] == password:
            session['user_id'] = user['id']
            session['user_name'] = user['firstName'] + ' ' + user['lastName']
            session['user_email'] = email
            session['user_role'] = user.get('role', 'user')
            # Update last login
            user['lastLogin'] = datetime.now().isoformat()
            save_users(users_db)
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

    @app.route('/register')
    def register():
        if 'user_id' in session:
            return redirect(url_for('index'))
        return render_template('register.html')

    @app.route('/register', methods=['POST'])
    def register_post():
        data = request.get_json()

        # Validate required fields
        required_fields = ['firstName', 'lastName', 'email', 'password', 'confirmPassword', 'phone', 'location']
        for field in required_fields:
            if not data.get(field, '').strip():
                return jsonify({'success': False, 'message': f'{field} is required'}), 400

        email = data['email'].strip().lower()
        password = data['password']
        confirm_password = data['confirmPassword']

        # Check if passwords match
        if password != confirm_password:
            return jsonify({'success': False, 'message': 'Passwords do not match'}), 400

        # Check password length
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters long'}), 400

        # Check if user already exists
        if email in users_db:
            return jsonify({'success': False, 'message': 'Email already registered'}), 409

        # Create new user
        user_id = str(len(users_db) + 1)
        user_data = {
            'id': user_id,
            'firstName': data['firstName'].strip(),
            'lastName': data['lastName'].strip(),
            'email': email,
            'password': password,  # In production, hash this password!
            'phone': data['phone'].strip(),
            'location': data['location'],
            'farmSize': data.get('farmSize', ''),
            'role': 'user',  # Default role
            'registrationDate': datetime.now().isoformat(),
            'lastLogin': None
        }

        users_db[email] = user_data
        save_users(users_db)

        # Don't auto-login after registration - redirect to login page
        return jsonify({'success': True, 'message': 'Registration successful'})

    @app.route('/logout')
    def logout():
        session.clear()
        return redirect(url_for('login'))

    # Protected Routes
    @app.route('/')
    @login_required
    def index():
        return render_template('index.html')

    @app.route('/voice-recognition')
    @login_required
    def voice_recognition_page():
        return render_template('voice_recognition.html')

    @app.route('/weather')
    @login_required
    def weather():
        return render_template('weather.html')

    @app.route('/plant-scan')
    @login_required
    def plant_scan():
        return render_template('plant_scan.html')

    @app.route('/voice-assistant')
    @login_required
    def voice_assistant():
        return render_template('voice_assistant.html')

    # Admin Routes
    @app.route('/admin')
    @admin_required
    def admin_dashboard():
        # Get user statistics
        total_users = len(users_db)
        active_users = sum(1 for user in users_db.values() if user.get('lastLogin'))
        recent_registrations = sum(1 for user in users_db.values()
                                   if user.get('registrationDate') and
                                   (datetime.now() - datetime.fromisoformat(user['registrationDate'])).days <= 30)

        # Get system stats (mock data for now)
        system_stats = {
            'total_users': total_users,
            'active_users': active_users,
            'recent_registrations': recent_registrations,
            'total_queries': 0,  # Would need to track this
            'active_sessions': 1 if 'user_id' in session else 0
        }

        return render_template('admin/dashboard.html', stats=system_stats, users=list(users_db.values()))

    @app.route('/admin/users')
    @admin_required
    def admin_users():
        return render_template('admin/users.html', users=list(users_db.values()))

    @app.route('/admin/analytics')
    @admin_required
    def admin_analytics():
        # Get user statistics
        total_users = len(users_db)
        # Mock analytics data
        analytics = {
            'user_growth': [1, 2, 3, total_users],
            'query_trends': [10, 15, 20, 25],
            'popular_features': ['voice_recognition', 'weather', 'plant_scan']
        }
        return render_template('admin/analytics.html', analytics=analytics, total_users=total_users)

    @app.route('/admin/settings')
    @admin_required
    def admin_settings():
        # Mock settings
        settings = {
            'app_name': 'AgriNathi',
            'debug_mode': app.config['DEBUG'] if 'DEBUG' in app.config else False,
            'max_upload_size': '10MB',
            'supported_languages': ['en', 'zu']
        }
        return render_template('admin/settings.html', settings=settings)

    @app.route('/admin/users/<user_id>', methods=['DELETE'])
    @admin_required
    def delete_user(user_id):
        # Find user by ID
        user_to_delete = None
        user_email = None
        for email, user in users_db.items():
            if user['id'] == user_id:
                user_to_delete = user
                user_email = email
                break

        if not user_to_delete:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        # Don't allow deleting self
        if session.get('user_id') == user_id:
            return jsonify({'success': False, 'message': 'Cannot delete your own account'}), 400

        # Remove user
        del users_db[user_email]
        save_users(users_db)

        return jsonify({'success': True, 'message': 'User deleted successfully'})

    @app.route('/admin/users/<user_id>/role', methods=['POST'])
    @admin_required
    def update_user_role(user_id):
        data = request.get_json()
        new_role = data.get('role')

        if new_role not in ['user', 'admin']:
            return jsonify({'success': False, 'message': 'Invalid role'}), 400

        # Find user by ID
        user_email = None
        for email, user in users_db.items():
            if user['id'] == user_id:
                user_email = email
                break

        if not user_email:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        users_db[user_email]['role'] = new_role
        save_users(users_db)

        return jsonify({'success': True, 'message': 'User role updated successfully'})

    @app.route('/voice-query', methods=['POST'])
    def voice_query():
        """
        Enhanced voice query endpoint with comprehensive error handling and logging.
        Processes audio input through the complete pipeline: Audio → Text → Response → Audio
        """
        request_start = datetime.now()
        client_ip = request.remote_addr
        print(f"[VOICE_QUERY] Request from {client_ip} at {request_start}")

        try:
            # Validate request data
            data = request.get_json()
            if not data:
                print("[VOICE_QUERY] No JSON data received")
                return jsonify({'error': 'Invalid JSON data'}), 400

            if 'audio' not in data:
                print("[VOICE_QUERY] No audio data in request")
                return jsonify({'error': 'No audio data provided'}), 400

            audio_base64 = data['audio']
            if not audio_base64 or len(audio_base64.strip()) == 0:
                print("[VOICE_QUERY] Empty audio data")
                return jsonify({'error': 'Empty audio data'}), 400

            print(f"[VOICE_QUERY] Processing audio data of length: {len(audio_base64)}")

            # Process the audio through enhanced pipeline
            result = voice_recognition.process_audio(audio_base64)

            request_end = datetime.now()
            processing_time = (request_end - request_start).total_seconds()

            # Add request metadata to response
            result['request_metadata'] = {
                'processing_time': processing_time,
                'timestamp': request_end.isoformat(),
                'client_ip': client_ip,
                'audio_size': len(audio_base64)
            }

            print(f"[VOICE_QUERY] Request completed in {processing_time:.2f} seconds")

            # Return appropriate HTTP status based on success
            status_code = 200 if result.get('success', False) else 500
            return jsonify(result), status_code

        except Exception as e:
            request_end = datetime.now()
            processing_time = (request_end - request_start).total_seconds()

            error_msg = f'Internal server error: {str(e)}'
            print(f"[VOICE_QUERY] Error after {processing_time:.2f} seconds: {error_msg}")

            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'request_metadata': {
                    'processing_time': processing_time,
                    'timestamp': request_end.isoformat(),
                    'client_ip': client_ip
                }
            }), 500

    @app.route('/test-voice')
    def test_voice():
        return jsonify({
            'message': 'Voice recognition system is active and ready.',
            'success': True
        })

    @app.route('/general-advice-audio')
    @login_required
    def general_advice_audio():
        """Get general farming advice with Zulu audio"""
        result = voice_recognition.get_advice_audio("general farming advice")
        return jsonify(result)

    @app.route('/text-to-speech', methods=['GET', 'POST'])
    @login_required
    def text_to_speech():
        """
        Enhanced text-to-speech endpoint with comprehensive validation and error handling.
        Converts text input to high-quality audio output.
        """
        if request.method == 'POST':
            request_start = datetime.now()
            user_email = session.get('user_email', 'unknown')
            print(f"[TEXT_TO_SPEECH] Request from user {user_email} at {request_start}")

            try:
                # Validate and parse request data
                data = request.get_json()
                if not data:
                    print("[TEXT_TO_SPEECH] No JSON data received")
                    return jsonify({'error': 'Invalid JSON data'}), 400

                text = data.get('text', '').strip()
                language_code = data.get('language_code', 'zu-ZA')
                voice_gender = data.get('voice_gender', 'NEUTRAL')

                # Comprehensive input validation
                if not text:
                    print("[TEXT_TO_SPEECH] No text provided")
                    return jsonify({'error': 'No text provided'}), 400

                if len(text) > 5000:  # Reasonable limit for TTS
                    print(f"[TEXT_TO_SPEECH] Text too long: {len(text)} characters")
                    return jsonify({'error': 'Text too long (maximum 5000 characters)'}), 400

                # Validate language code
                supported_languages = ['zu-ZA', 'en-US', 'en-GB']
                if language_code not in supported_languages:
                    print(f"[TEXT_TO_SPEECH] Unsupported language: {language_code}")
                    return jsonify({'error': f'Unsupported language. Supported: {", ".join(supported_languages)}'}), 400

                # Validate voice gender
                supported_genders = ['MALE', 'FEMALE', 'NEUTRAL']
                if voice_gender.upper() not in supported_genders:
                    print(f"[TEXT_TO_SPEECH] Unsupported voice gender: {voice_gender}")
                    return jsonify({'error': f'Unsupported voice gender. Supported: {", ".join(supported_genders)}'}), 400

                voice_gender = voice_gender.upper()

                print(f"[TEXT_TO_SPEECH] Synthesizing speech for '{text[:50]}...' in {language_code} with {voice_gender} voice")

                # Synthesize speech with enhanced error handling
                audio_content = tts_service.synthesize_speech(text, language_code, voice_gender)

                if audio_content is None:
                    print("[TEXT_TO_SPEECH] Speech synthesis returned None")
                    return jsonify({'error': 'Speech synthesis failed - no audio generated'}), 500

                audio_size = len(audio_content)
                print(f"[TEXT_TO_SPEECH] Synthesis successful: {audio_size} bytes")

                # Validate audio content
                if audio_size < 100:  # Basic validation for audio file
                    print(f"[TEXT_TO_SPEECH] Audio content too small: {audio_size} bytes")
                    return jsonify({'error': 'Generated audio is invalid or too small'}), 500

                # Encode to base64
                audio_base64 = base64.b64encode(audio_content).decode('utf-8')

                request_end = datetime.now()
                processing_time = (request_end - request_start).total_seconds()

                print(f"[TEXT_TO_SPEECH] Request completed in {processing_time:.2f} seconds")

                return jsonify({
                    'success': True,
                    'audio_content': audio_base64,
                    'text': text,
                    'language_code': language_code,
                    'voice_gender': voice_gender,
                    'audio_size': audio_size,
                    'processing_time': processing_time,
                    'timestamp': request_end.isoformat()
                })

            except Exception as e:
                request_end = datetime.now()
                processing_time = (request_end - request_start).total_seconds()

                error_msg = f'Text-to-speech failed: {str(e)}'
                print(f"[TEXT_TO_SPEECH] Error after {processing_time:.2f} seconds: {error_msg}")

                return jsonify({
                    'error': 'Text-to-speech conversion failed',
                    'details': str(e),
                    'processing_time': processing_time,
                    'timestamp': request_end.isoformat()
                }), 500

        return render_template('text_to_speech.html')

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)