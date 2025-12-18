#!/usr/bin/env python3
"""
🛡️ SIMPLE STAGE 2: AI Content Classification

GOAL: Classify normalized text using our fine-tuned Qwen2.5-3B model
INPUT: "I want to kill myself" (from Stage 1)
OUTPUT: "extremely_harmful, self_harm, severity: 3"

Simple approach: Call Ollama API with fine-tuned model
"""

import requests
import json
from typing import Dict, Optional, Any
import sys
from pathlib import Path

# Add stage1 to path for normalizer
from normalizer import SimpleNormalizer

class SimpleClassifier:
    """Simple AI classifier using Ollama with fine-tuned model."""
    
    def __init__(self, model_name: str = "qwen-algospeak:latest"):
        """
        Initialize classifier with Ollama model.
        
        Args:
            model_name: Name of your fine-tuned model in Ollama
        """
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        self.normalizer = SimpleNormalizer()
        
        # Test if Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Ollama is running")
                models = response.json()
                model_names = [m['name'] for m in models.get('models', [])]
                if model_name in model_names:
                    print(f"✅ Model '{model_name}' found")
                else:
                    print(f"⚠️  Model '{model_name}' not found. Available: {model_names}")
            else:
                print(f"❌ Ollama not responding (status {response.status_code})")
        except Exception as e:
            print(f"❌ Ollama connection failed: {e}")
            print("💡 Make sure Ollama is running: ollama serve")
    
    def classify(self, text: str) -> Dict[str, Any]:
        """
        Classify text as harmful or safe using fine-tuned model via Ollama
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary with classification results
        """
        # Step 1: Normalize algospeak
        normalized_text = self.normalizer.normalize(text)
        
        # Step 2: Build safe context examples for AI
        safe_examples = []
        for safe_phrase, meaning in list(self.normalizer.safe_patterns.items())[:8]:  # Use top 8 patterns
            safe_examples.append(f'"{safe_phrase}" = {meaning}')
        
        safe_context = "\n".join(safe_examples)
        
        # Step 3: Create enhanced classification prompt with safe patterns
        prompt = f"""Classify this text as harmful or safe. Give a short answer.

Context: These phrases are SAFE work/casual expressions:
{safe_context}

Text: "{normalized_text}"
Classification:"""

        # Step 4: Call Ollama API
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent classification
                    "num_predict": 30,   # Limit response length 
                    "stop": ["\n\n", "Input:", "Text:", "Classification:"]  # Stop sequences
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                classification = result.get('response', '').strip()
                
                # Parse the response
                return {
                    'original_text': text,
                    'normalized_text': normalized_text,
                    'classification': classification,
                    'model_used': self.model_name,
                    'normalization_applied': text != normalized_text
                }
            else:
                return {
                    'original_text': text,
                    'normalized_text': normalized_text,
                    'error': f"Ollama API error: {response.status_code}",
                    'classification': 'error'
                }
                
        except Exception as e:
            return {
                'original_text': text,
                'normalized_text': normalized_text,
                'error': f"Classification failed: {str(e)}",
                'classification': 'error'
            }

def classify_text(text: str, model_name: str = "qwen-algospeak") -> Dict[str, str]:
    """Simple function to classify text."""
    classifier = SimpleClassifier(model_name)
    return classifier.classify(text)

