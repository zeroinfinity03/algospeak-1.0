#!/usr/bin/env python3
"""
🧪 SIMPLE MODEL TEST - Just check if the model works

This script just tests if the model can respond to basic questions:
- "Hi how are you?"
- "What is 2 + 2?"
- Simple conversation test

Usage: python simple_test.py
"""

import requests
import json

def test_ollama_basic():
    """Test if Ollama service is running"""
    print("🔍 Checking Ollama service...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            models = response.json()
            model_list = [m['name'] for m in models.get('models', [])]
            print(f"📋 Available models: {model_list}")
            return True, model_list
        else:
            print(f"❌ Ollama service not responding")
            return False, []
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("💡 Make sure to run: ollama serve")
        return False, []

def simple_chat_test(model_name, question):
    """Send a simple question to the model"""
    print(f"\n🤖 Testing: '{question}'")
    
    try:
        payload = {
            "model": model_name,
            "prompt": question,
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower temperature for more focused responses
                "num_predict": 50,   # Limit to 50 tokens max
                "stop": ["\n\n", "Question:", "Q:", "User:"]  # Stop sequences
            }
        }
        
        response = requests.post("http://localhost:11434/api/generate", 
                               json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '').strip()
            print(f"✅ Model response: {answer}")
            return True
        else:
            print(f"❌ API error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
        return False

def main():
    """Run simple model tests"""
    print("🧪 SIMPLE MODEL TEST")
    print("=" * 40)
    
    # Check if Ollama is running
    service_ok, models = test_ollama_basic()
    if not service_ok:
        return
    
    # Find our model
    target_model = "qwen-algospeak:latest"
    if target_model not in models:
        print(f"❌ Model '{target_model}' not found")
        print("💡 Available models:", models)
        print("\n🔧 To create the model:")
        print("1. echo 'FROM ./quantized_model/unsloth.Q4_K_M.gguf' > Modelfile")
        print("2. ollama create qwen-algospeak -f Modelfile")
        return
    
    print(f"\n✅ Found model: {target_model}")
    
    # Simple conversation tests with shorter prompts
    test_questions = [
        "Hi, how are you?",
        "What is 2 + 2?", 
        "Name one color.",
        "Say hello."
    ]
    
    print(f"\n🧪 Testing {len(test_questions)} simple questions...")
    passed = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Test {i}/{len(test_questions)} ---")
        if simple_chat_test(target_model, question):
            passed += 1
        else:
            print("❌ Test failed")
    
    print(f"\n🎯 Results: {passed}/{len(test_questions)} tests passed")
    
    if passed == len(test_questions):
        print("🎉 MODEL IS WORKING! Now testing algospeak classification...")
        
        # Test actual algospeak classification
        print("\n🔍 Testing algospeak classification:")
        algospeak_prompt = """Classify this text as harmful or safe. Give a short answer.

Text: "I want to unalive myself"
Classification:"""
        
        if simple_chat_test(target_model, algospeak_prompt):
            print("✅ Algospeak classification works!")
        else:
            print("❌ Algospeak classification failed")
    else:
        print("⚠️ Some tests failed - model may have issues")

if __name__ == "__main__":
    main() 

