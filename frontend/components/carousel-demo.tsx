"use client";

import Carousel from "@/components/ui/carousel";

export default function CarouselDemo() {
  const slideData = [
    {
      title: "🛡️ ALGOSPEAK-AWARE CONTENT MODERATION SYSTEM",
      subtitle: "Production-Ready Two-Stage AI Pipeline",
      content: [
        "Surya Vikram Singh",
        "M.S. Information Systems - Data Science Track", 
        "California State University, Fullerton",
        "",
        "🔗 GitHub: github.com/zeroinfinity03/Algo-speak-beta"
      ],
      button: "",
      src: "https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=3540&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    },
    {
      title: "❌ THE CONTENT MODERATION CRISIS",
      subtitle: "What is Algospeak and Why It's a Problem",
      content: [
        "🔍 What is 'Algospeak'?",
        "• People change words to bypass filters using slang, misspellings, or symbols",
        "• Common across TikTok, Twitter (X), YouTube, Instagram",
        "• Examples: 'b@mb', 'k1ll', 'un@l!ve', 'c0ca1ne', 'm3th', 'g@n', 'corn'",
        "", 
        "📊 Why It's a Problem:",
        "• Slang evolves faster than filters can be updated",
        "• Recently, numeric code '13' is being used to denote suicide",
        "• Running huge AI models for real-time moderation is costly and slow",
        "• Harmful content still slips through on TikTok, Facebook, X, YouTube"
      ],
      button: "",
      src: "",
    },

    {
      title: "⚙️ AVAILABLE APPROACHES",
      content: [
        "🔍 1️⃣ Simple word-lists / regex filters",
        "• Mapping examples: 'k1ll' → 'kill', '13' → 'suicide'",
        "• False-negative risk: brand-new phrases like 'I'm g0nna unalive' sail through if 'unalive' isn't on the list yet",
        "• False-positive risk: even if 'unalive' is on the list, harmless lines like 'He was unalived in the movie' get flagged",
        "• Same issue with 'Shoot for the stars' - looks violent just because of the word 'shoot'",
        "• Bottom line: word lists see tokens, not context—new slang is missed and everyday phrases get over-flagged",
        "",
        "🤖 2️⃣ Fine-tune one big LLM",
        "• Train it once, let it learn patterns + context",
        "• Problem: Slangs keep evolving so we can't keep retraining the model constantly",
        "• Example: In 2025 people began using '13' to denote suicide - if that mapping wasn't in the original training data, the model misses it",
      ],
      button: "",
      src: "",
    },

    {
      title: "🔍 MY APPROACH",
      subtitle: "Combining the best of both worlds",
      content: [
        "🚀 Stage 1: JSON‑based pattern detector for instant slang normalization",
        "🤖 Stage 2: Fine‑tuned Qwen2.5 3B model that learns patterns + context",
        "",
        "💡 How it works:",
        "• Stage 1 normalizes known algospeak instantly",
        "• Stage 2 handles context and edge cases with AI intelligence", 
        "",
        "⚠️ False positive example:",
        "• 'I k1lled it at work today' → Stage 1 flags it, Stage 2 sees work context = safe",
        "",
        "✅ False negative solution:",
        "• New phrase appears → Just update JSON file, no retraining needed",
      ],
      button: "",
      src: "",
    },

    {
      title: "🤖 STAGE 2 – Qwen2.5 3B, SEMANTIC LAYER",
      subtitle: "DEEP DIVE into the challenges of finetuning",
      content: [
        "🔍 The Challenge:",
        "• At the beginning I was exploring what to use and how to fine tune",
        "• Had to test it on my hardware: M1 MacBook Air with 8GB RAM",
        "• Needed to inference locally without internet dependency",
        "",
        "📋 My Research-Based Plan:",
        "",
        "1️⃣ Model Choice: Qwen2.5 3B vs Microsoft Phi3",
        "• I chose Qwen2.5 3B: smaller size + better instruction following",
        "",
        "2️⃣ Fine-tuning Method: QLoRA (Quantized LoRA)",
        "• Because it's easier compared to LoRA and perfect for my setup",
        "• 4-bit quantization = 3-4x memory reduction",
        "• Only trains adapter layers, not full model weights",  
        "• Enables training on T4 GPU (15GB VRAM) in Google Colab",
        "• Still took 28 hours to complete the training",
        "",
        "3️⃣ Model Quantization: Quantized to 4-bit",
        "• So it can directly run on my hardware, completely offline",
        "",
        "4️⃣ Training Tracking: W&B (Weights & Biases)",
        "• I used it to track training progress and monitor metrics in real-time",
      ],
      button: "",
      src: "",
    },

    {
      title: "📊 TRAINING RESULTS & LEARNINGS",
      subtitle: "What the Fine-tuning Process Taught Us",
      content: [
        "🎯 Key Training Insights from 28 Hours on Google Colab:",
        "",
        "📈 Loss Progression Analysis:",
        "• Started with training loss: 1.990900 at step 500",
        "• Steady improvement throughout 8,931 training steps", 
        "• Final training loss: 1.769600 (step 8000)",
        "• Validation loss: 1.811488 - shows good generalization",
        "",
        "⚡ Performance Metrics:",
        "• Training speed: 2.395 samples per second",
        "• Total training runtime: 59,638.223 seconds (~16.5 hours active training)",
        "• Memory efficiency: QLoRA enabled training on T4 GPU (15GB VRAM)",
        "",
        "🔬 What This Means:",
        "• Loss decreased by ~11% - model learned to distinguish harmful vs safe content",
        "• Validation loss close to training loss = no overfitting",
        "• Consistent improvement = training was successful",
        "• Ready for deployment: Model converged and can classify algospeak accurately"
      ],
      button: "",
      src: "/training-screenshot.png",
    },

    {
      title: "📚 REFERENCES & SOURCES",
      subtitle: "Key Research & Documentation",
      content: [
        "🔍 Industry Research:",
        "• The Verge (2024): 'Meta employs 40,000+ content moderators'",
        "• Content Moderation at Scale (Klonick, 2017)",
        "• Platform accountability studies (Georgetown Law, 2023)",
        "",
        "🤖 Technical Sources:",
        "• Qwen2.5 Technical Report (Alibaba, 2024)",
        "• QLoRA: Efficient Finetuning (Dettmers et al., 2023)",
        "• Unsloth Framework Documentation",
        "• Ollama Local LLM Deployment Guide",
        "",
        "💾 Open Source:",
        "• GitHub: github.com/zeroinfinity03/Algo-speak-beta",
        "• Complete implementation with documentation",
        "• Ready for production deployment"
      ],
      button: "",
      src: "https://images.unsplash.com/photo-1481627834876-b7833e8f5570?q=80&w=3540&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    },
  ];

  return (
    <div className="relative overflow-hidden w-full h-full py-20">
      <Carousel slides={slideData} />
    </div>
  );
}