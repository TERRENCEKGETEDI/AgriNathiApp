import json
import os
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class AgricultureMLService:
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self._fit_vectorizer()

    def _load_knowledge_base(self) -> Dict[str, List[Dict]]:
        """Load agricultural knowledge base from JSON files"""
        knowledge_base = {
            'diseases': [],
            'pests': [],
            'fertilizers': [],
            'watering': [],
            'planting': [],
            'weather': [],
            'soil': [],
            'general': []
        }

        # Load pest and disease information
        pest_disease_file = 'data/pest_disease_info.json'
        if os.path.exists(pest_disease_file):
            try:
                with open(pest_disease_file, 'r', encoding='utf-8') as f:
                    pest_data = json.load(f)
                    for item in pest_data:
                        if 'type' in item and item['type'] == 'disease':
                            knowledge_base['diseases'].append(item)
                        elif 'type' in item and item['type'] == 'pest':
                            knowledge_base['pests'].append(item)
            except Exception as e:
                print(f"Error loading pest/disease data: {e}")

        # Add predefined knowledge
        knowledge_base['diseases'].extend([
            {
                'name': 'Leaf Blight',
                'symptoms': ['brown spots on leaves', 'yellowing leaves', 'leaf drop'],
                'causes': ['fungal infection', 'high humidity', 'poor air circulation'],
                'solutions': ['Remove affected leaves', 'Improve air circulation', 'Apply copper-based fungicide', 'Avoid overhead watering']
            },
            {
                'name': 'Root Rot',
                'symptoms': ['wilting plants', 'yellow leaves', 'soft rotten roots'],
                'causes': ['overwatering', 'poor drainage', 'fungal pathogens'],
                'solutions': ['Improve drainage', 'Reduce watering frequency', 'Use well-draining soil', 'Apply fungicide to soil']
            }
        ])

        knowledge_base['pests'].extend([
            {
                'name': 'Aphids',
                'symptoms': ['sticky leaves', 'distorted growth', 'ants on plants'],
                'solutions': ['Spray with soapy water', 'Introduce ladybugs', 'Use neem oil', 'Remove heavily infested parts']
            },
            {
                'name': 'Cutworms',
                'symptoms': ['seedlings cut at base', 'plants falling over'],
                'solutions': ['Use collars around seedlings', 'Hand pick at night', 'Apply beneficial nematodes', 'Use organic pesticides']
            }
        ])

        knowledge_base['fertilizers'].extend([
            {
                'name': 'NPK Fertilizer',
                'type': 'balanced',
                'usage': ['Apply during growing season', 'Mix with soil', 'Water after application'],
                'benefits': ['Promotes overall plant growth', 'Improves fruit/seed production', 'Strengthens root system']
            },
            {
                'name': 'Organic Compost',
                'type': 'organic',
                'usage': ['Mix into topsoil', 'Apply around plants', 'Use as mulch'],
                'benefits': ['Improves soil structure', 'Provides slow-release nutrients', 'Enhances microbial activity']
            }
        ])

        knowledge_base['watering'].extend([
            {
                'advice': 'Water deeply but infrequently to encourage deep root growth',
                'tips': ['Water early morning or evening', 'Check soil moisture before watering', 'Avoid wetting leaves', 'Use drip irrigation when possible']
            }
        ])

        knowledge_base['planting'].extend([
            {
                'crop': 'Maize',
                'season': 'Summer rainfall areas: October-November',
                'spacing': '90cm between rows, 30-45cm between plants',
                'depth': '3-5cm deep',
                'tips': ['Plant after soil temperature reaches 10°C', 'Ensure good seed-to-soil contact', 'Plant in blocks for better pollination']
            }
        ])

        return knowledge_base

    def _fit_vectorizer(self):
        """Fit the vectorizer with all text from knowledge base"""
        all_texts = []
        for category, items in self.knowledge_base.items():
            for item in items:
                # Extract all text fields from each item
                for key, value in item.items():
                    if isinstance(value, str):
                        all_texts.append(value)
                    elif isinstance(value, list):
                        all_texts.extend([str(v) for v in value])

        if all_texts:
            self.vectorizer.fit(all_texts)

    def find_solution(self, query: str) -> Optional[Dict]:
        """
        Find the most relevant agricultural solution for a given query
        """
        query = query.lower().strip()

        # Direct keyword matching first
        direct_match = self._direct_keyword_match(query)
        if direct_match:
            return direct_match

        # ML-based similarity matching
        return self._ml_similarity_match(query)

    def _direct_keyword_match(self, query: str) -> Optional[Dict]:
        """Direct keyword matching for common agricultural terms"""
        keywords = {
            'disease': 'diseases',
            'diseased': 'diseases',
            'sick': 'diseases',
            'infection': 'diseases',
            'fungus': 'diseases',
            'blight': 'diseases',
            'rot': 'diseases',
            'pest': 'pests',
            'insect': 'pests',
            'bug': 'pests',
            'aphid': 'pests',
            'worm': 'pests',
            'fertilizer': 'fertilizers',
            'manure': 'fertilizers',
            'compost': 'fertilizers',
            'water': 'watering',
            'watering': 'watering',
            'irrigation': 'watering',
            'plant': 'planting',
            'seed': 'planting',
            'sow': 'planting',
            'maize': 'planting',
            'corn': 'planting'
        }

        for keyword, category in keywords.items():
            if keyword in query:
                items = self.knowledge_base.get(category, [])
                if items:
                    # Return the first relevant item
                    return {
                        'category': category,
                        'solution': items[0],
                        'confidence': 0.8
                    }

        return None

    def _ml_similarity_match(self, query: str) -> Optional[Dict]:
        """Use ML similarity matching to find relevant solutions"""
        try:
            # Vectorize the query
            query_vector = self.vectorizer.transform([query])

            best_match = None
            best_score = 0.0

            # Check similarity against all items in knowledge base
            for category, items in self.knowledge_base.items():
                for item in items:
                    # Create text representation of the item
                    item_texts = []
                    for key, value in item.items():
                        if isinstance(value, str):
                            item_texts.append(value)
                        elif isinstance(value, list):
                            item_texts.extend([str(v) for v in value])

                    item_text = ' '.join(item_texts)
                    item_vector = self.vectorizer.transform([item_text])

                    # Calculate similarity
                    similarity = cosine_similarity(query_vector, item_vector)[0][0]

                    if similarity > best_score and similarity > 0.1:  # Minimum threshold
                        best_score = similarity
                        best_match = {
                            'category': category,
                            'solution': item,
                            'confidence': float(similarity)
                        }

            return best_match

        except Exception as e:
            print(f"ML similarity matching error: {e}")
            return None

    def generate_advice_text(self, query: str) -> str:
        """Generate dynamic advice text based on user query with random elements"""
        import random

        # Analyze the query and generate relevant advice
        query_lower = query.lower()

        # Generate a random advice number for uniqueness
        advice_number = random.randint(1000, 9999)

        # Disease-related queries
        if any(word in query_lower for word in ['disease', 'diseased', 'sick', 'infection', 'fungus', 'blight', 'rot', 'zifo']):
            treatments = [
                "Remove affected leaves and improve air circulation",
                "Apply copper-based fungicide spray",
                "Use organic neem oil solution",
                "Ensure proper plant spacing to reduce humidity"
            ]
            treatment = random.choice(treatments)
            return f"Advice #{advice_number}: For plant diseases, {treatment}. Monitor regularly and consult extension services if condition worsens."

        # Pest-related queries
        elif any(word in query_lower for word in ['pest', 'insect', 'bug', 'aphid', 'worm', 'nambuzane']):
            controls = [
                "Use soapy water spray to remove pests",
                "Introduce beneficial ladybugs to control aphids",
                "Apply diatomaceous earth around plants",
                "Use row covers to protect young plants"
            ]
            control = random.choice(controls)
            return f"Advice #{advice_number}: For pest control, {control}. Regular monitoring is essential for effective management."

        # Watering-related queries
        elif any(word in query_lower for word in ['water', 'watering', 'nisela']):
            watering_tips = [
                "Water deeply but infrequently to encourage deep roots",
                "Water early morning to reduce evaporation",
                "Check soil moisture 2 inches deep before watering",
                "Use drip irrigation for consistent moisture"
            ]
            tip = random.choice(watering_tips)
            return f"Advice #{advice_number}: For watering, {tip}. Proper watering prevents both drought stress and root rot."

        # Fertilizer-related queries
        elif any(word in query_lower for word in ['fertilizer', 'manure', 'compost', 'umanyolo']):
            fertilizer_advice = [
                "Use balanced NPK fertilizer during growing season",
                "Apply organic compost around plant bases",
                "Test soil pH before fertilizing",
                "Use slow-release fertilizers for steady nutrition"
            ]
            advice = random.choice(fertilizer_advice)
            return f"Advice #{advice_number}: For fertilization, {advice}. Regular soil testing ensures optimal nutrient balance."

        # Planting-related queries
        elif any(word in query_lower for word in ['plant', 'seed', 'sow', 'imbewu']):
            planting_tips = [
                "Plant during correct season for best germination",
                "Ensure proper seed-to-soil contact",
                "Space plants according to variety requirements",
                "Keep soil consistently moist until germination"
            ]
            tip = random.choice(planting_tips)
            return f"Advice #{advice_number}: For planting, {tip}. Proper planting techniques maximize crop success."

        # Weather-related queries
        elif any(word in query_lower for word in ['weather', 'rain', 'sun', 'isimo sezulu']):
            weather_advice = [
                "Monitor weather forecasts for planting decisions",
                "Protect crops from frost using covers",
                "Prepare drainage for heavy rain periods",
                "Use windbreaks in exposed areas"
            ]
            advice = random.choice(weather_advice)
            return f"Advice #{advice_number}: For weather considerations, {advice}. Weather adaptation improves crop resilience."

        # Soil-related queries
        elif any(word in query_lower for word in ['soil', 'ground', 'earth', 'umhlaba']):
            soil_tips = [
                "Test soil pH regularly for optimal fertility",
                "Add organic matter to improve soil structure",
                "Practice crop rotation to maintain soil health",
                "Use cover crops to prevent erosion"
            ]
            tip = random.choice(soil_tips)
            return f"Advice #{advice_number}: For soil management, {tip}. Healthy soil is the foundation of successful farming."

        # General farming queries
        else:
            general_advice = [
                "Practice sustainable farming methods",
                "Monitor crops regularly for early problem detection",
                "Maintain soil health through organic practices",
                "Seek local extension services for specific guidance",
                "Keep detailed farming records for better planning",
                "Use integrated pest management approaches"
            ]
            advice = random.choice(general_advice)
            return f"Advice #{advice_number}: {advice}. Consistent good practices lead to better farming outcomes."