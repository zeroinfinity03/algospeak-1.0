#!/bin/bash

# =========================================================
# Algospeak Backend Setup Script
# Run this from the backend directory: ./setup.sh
# =========================================================

echo "🛡️  Algospeak Content Moderation - Backend Setup"
echo "================================================="
echo ""

# Check if we're in the backend directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Please run this script from the backend directory"
    echo "   cd backend && ./setup.sh"
    exit 1
fi

# Step 1: Check if Ollama is installed
echo "📋 Step 1: Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Installing with Homebrew..."
    brew install ollama
else
    echo "✅ Ollama is installed"
fi

# Step 2: Check if Ollama is running, start if not
echo ""
echo "📋 Step 2: Starting Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama is already running"
else
    echo "🚀 Starting Ollama in background..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    echo "   Ollama started with PID: $OLLAMA_PID"
    
    # Wait for Ollama to be ready
    echo "   Waiting for Ollama to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "✅ Ollama is ready"
            break
        fi
        sleep 1
    done
fi

# Step 3: Register the model with Ollama
echo ""
echo "📋 Step 3: Registering model with Ollama..."
if ollama list | grep -q "qwen-algospeak"; then
    echo "✅ Model 'qwen-algospeak' already exists"
else
    if [ -f "Modelfile" ] && [ -f "quantized_model/unsloth.Q4_K_M.gguf" ]; then
        echo "🔧 Creating model from Modelfile..."
        ollama create qwen-algospeak -f Modelfile
        echo "✅ Model created successfully"
    else
        echo "❌ Error: Modelfile or quantized_model/unsloth.Q4_K_M.gguf not found"
        echo "   Make sure you have the model file in the correct location"
        exit 1
    fi
fi

# Step 4: Install Python dependencies
echo ""
echo "📋 Step 4: Installing Python dependencies..."
if command -v uv &> /dev/null; then
    uv sync
    echo "✅ Python dependencies installed"
else
    echo "❌ uv not found. Install with: pip install uv"
    exit 1
fi

# Step 5: Start the FastAPI server
echo ""
echo "📋 Step 5: Starting FastAPI server..."
echo "================================================="
echo ""
echo "🚀 Backend is starting on http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
echo "❤️  Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py
