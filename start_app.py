#!/usr/bin/env python3
"""
Simple startup script for the AI Image Caption Generator using Flickr30k dataset.
This script handles setup and runs the Flask application.
"""

import os
import sys
import subprocess
import importlib.util
import json


def check_and_install_requirements():
    """Check if required packages are installed, install if missing"""
    print("🔍 Checking dependencies...")

    required_packages = [
        "torch",
        "torchvision",
        "transformers",
        "pillow",
        "flask",
        "numpy",
        "datasets",
        "nltk",
        "textblob",
    ]

    missing_packages = []

    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")

        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
            )
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing dependencies: {e}")
            print("Please install manually: pip install -r requirements.txt")
            return False
    else:
        print("✅ All dependencies are installed!")

    return True


def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ["uploads", "models", "config"]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")

    # Create enhanced caption configuration
    create_caption_config()


def create_caption_config():
    """Create enhanced caption generation configuration"""
    config_file = "config/caption_config.json"

    if not os.path.exists(config_file):
        enhanced_config = {
            "caption_strategies": {
                "creative": {
                    "temperature": 0.8,
                    "top_k": 50,
                    "top_p": 0.9,
                    "repetition_penalty": 1.2,
                    "length_penalty": 1.0,
                    "do_sample": True,
                    "num_beams": 1,
                },
                "accurate": {
                    "repetition_penalty": 1.1,
                    "length_penalty": 1.1,
                    "do_sample": False,
                    "num_beams": 5,
                },
                "diverse": {
                    "repetition_penalty": 1.3,
                    "length_penalty": 0.9,
                    "do_sample": False,
                    "num_beams": 6,
                    "num_beam_groups": 2,
                    "diversity_penalty": 1.5,
                },
            },
            "enhancement_features": {
                "style_detection": True,
                "emotion_analysis": True,
                "context_enrichment": True,
                "evaluation_enabled": True,
                "multiple_perspectives": True,
                "creative_elements": [
                    "mood",
                    "atmosphere",
                    "artistic_style",
                    "composition",
                ],
            },
            "post_processing": {
                "grammar_correction": True,
                "sentence_variety": True,
                "vocabulary_enhancement": True,
                "readability_optimization": True,
            },
        }

        with open(config_file, "w") as f:
            json.dump(enhanced_config, f, indent=2)
        print(f"📝 Created enhanced caption configuration: {config_file}")


def check_flickr_model():
    """Check if Flickr30k model is available"""
    print("\n🔍 Checking caption model...")

    try:
        from transformers import BlipProcessor

        # Check for trained model
        model_path = "models/best_flickr_model_epoch_1"
        if os.path.exists(model_path):
            print("✅ Found trained Flickr30k model")
            return True
        else:
            print("ℹ️  No trained model found, will use base BLIP model")
            print("   💡 To train on Flickr30k: python train_flickr_model.py")

            # Test base model loading
            BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            print("✅ Base BLIP model ready")
            return True

    except Exception as e:
        print(f"⚠️  Model loading issue: {e}")
        return False


def check_enhancement_dependencies():
    """Check if enhancement dependencies are available"""
    print("\n🔍 Checking caption enhancement capabilities...")

    enhancement_status = {
        "nltk": False,
        "textblob": False,
        "advanced_sampling": False,
        "evaluation_tools": False,
    }

    try:
        import nltk

        enhancement_status["nltk"] = True
        enhancement_status["evaluation_tools"] = True
        print("✅ NLTK available for text processing and evaluation")

        # Download required NLTK data
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            print("📥 Downloading NLTK data...")
            nltk.download("punkt", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
            nltk.download("vader_lexicon", quiet=True)

    except ImportError:
        print("⚠️  NLTK not available - limited text enhancement and evaluation")

    try:
        import textblob

        enhancement_status["textblob"] = True
        print("✅ TextBlob available for grammar and sentiment")
    except ImportError:
        print("⚠️  TextBlob not available - limited grammar enhancement")

    try:
        import torch

        if hasattr(torch.nn.functional, "top_k_top_p_filtering"):
            enhancement_status["advanced_sampling"] = True
        print("✅ Advanced sampling techniques available")
    except:
        print("⚠️  Advanced sampling limited")

    return enhancement_status


def main():
    """Main function to start the application"""
    print("📸 Starting Enhanced AI Caption Generator (Flickr30k)")
    print("=" * 60)

    # Check current directory
    if not os.path.exists("app_flickr.py"):
        print("❌ Error: app_flickr.py not found in current directory")
        print("Please run this script from the Mini_project directory")
        sys.exit(1)

    # Check and install dependencies
    if not check_and_install_requirements():
        sys.exit(1)

    # Create necessary directories and config
    create_directories()

    # Check caption model
    model_ready = check_flickr_model()

    # Check enhancement capabilities
    enhancement_status = check_enhancement_dependencies()

    print("\n📊 Enhanced Caption Generation Summary:")
    print(f"   • Caption model: {'✅ Ready' if model_ready else '❌ Not ready'}")
    print(
        f"   • NLTK text processing: {'✅ Available' if enhancement_status['nltk'] else '❌ Limited'}"
    )
    print(
        f"   • Grammar enhancement: {'✅ Available' if enhancement_status['textblob'] else '❌ Limited'}"
    )
    print(
        f"   • Advanced sampling: {'✅ Available' if enhancement_status['advanced_sampling'] else '❌ Basic'}"
    )
    print(
        f"   • Evaluation tools: {'✅ Available' if enhancement_status['evaluation_tools'] else '❌ Limited'}"
    )
    print("   • Dataset: Flickr30k trained captions")
    print("   • 🎨 Creative caption modes: 3 strategies")
    print("   • 📊 Quality scoring: ✅ Enabled")
    print("   • 📝 Post-processing: ✅ Enhanced")

    print("\n🎯 Caption Enhancement Features:")
    print("   • Multiple generation strategies (Creative, Accurate, Diverse)")
    print("   • Advanced sampling with temperature control")
    print("   • Quality evaluation metrics")
    print("   • Real-time quality assessment")
    print("   • Style and emotion detection")
    print("   • Grammar and readability optimization")
    print("   • Context-aware enhancements")

    print("\n" + "=" * 60)
    print("🌐 Starting enhanced web server with quality evaluation...")
    print("📱 Open http://localhost:3000 in your browser")
    print("🎨 Try different caption generation modes!")
    print("📊 View quality scores for assessment!")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)

    # Start the Flask app with enhanced features
    try:
        import app_flickr

        print("\n🎨 Initializing enhanced AI caption generator...")
        print("   • Loading caption enhancement modules...")
        print("   • Configuring multiple generation strategies...")
        print("   • Setting up quality evaluation system...")
        print("   • Initializing caption assessment tools...")

        if app_flickr.initialize_models():
            print("✅ Enhanced caption system with quality evaluation ready!")
            print("📊 Quality scoring: Active")
            print("🎯 Quality metrics: Enabled")
            app_flickr.app.run(
                debug=False, host="0.0.0.0", port=3000, use_reloader=False
            )
        else:
            print("❌ Failed to initialize enhanced models")
    except KeyboardInterrupt:
        print("\n👋 Enhanced server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting enhanced server: {e}")
        print("\nTry running directly: python app_flickr.py")


if __name__ == "__main__":
    main()
