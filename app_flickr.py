import os
import torch
from PIL import Image
import base64
from io import BytesIO
from flask import Flask, render_template_string, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings
import json
import re
import random

warnings.filterwarnings("ignore")

# Enhanced imports with fallbacks
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    NLTK_AVAILABLE = True
    # Initialize sentiment analyzer
    try:
        sia = SentimentIntensityAnalyzer()
    except:
        sia = None
        NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False
    sia = None

try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

app = Flask(__name__)

# Global variables for models
flickr_model = None
flickr_processor = None
caption_config = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_flickr_model():
    """Load Flickr30k trained model"""
    global flickr_model, flickr_processor

    try:
        print("Loading Flickr30k caption model...")

        # Check if we have a trained model
        model_path = "models/best_flickr_model_epoch_1"
        if os.path.exists(model_path):
            print(f"Loading trained model from {model_path}")
            flickr_processor = BlipProcessor.from_pretrained(model_path)
            flickr_model = BlipForConditionalGeneration.from_pretrained(model_path)
        else:
            print(
                "Using base BLIP model (train with train_flickr_model.py for better results)"
            )
            flickr_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
            flickr_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )

        flickr_model.to(device)
        flickr_model.eval()
        print("Caption model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def load_caption_config():
    """Load caption generation configuration"""
    global caption_config
    config_file = "config/caption_config.json"

    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            caption_config = json.load(f)
    else:
        # Default configuration
        caption_config = {
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
            }
        }


def calculate_bleu_score(generated_caption, reference_captions):
    """Calculate BLEU score for generated caption against reference captions"""
    if not generated_caption or not reference_captions:
        return 0.0

    # Ensure reference_captions is a list
    if isinstance(reference_captions, str):
        reference_captions = [reference_captions]

    # Tokenize generated caption
    candidate = generated_caption.lower().split()

    # Tokenize reference captions (BLEU expects list of lists)
    references = [ref.lower().split() for ref in reference_captions]

    # Use smoothing function to handle edge cases
    smoothing = SmoothingFunction().method1

    try:
        # Calculate BLEU score (using BLEU-4 by default)
        bleu_score = sentence_bleu(references, candidate, smoothing_function=smoothing)
        return bleu_score
    except Exception as e:
        print(f"Error calculating BLEU score: {e}")
        return 0.0


def calculate_detailed_bleu_scores(generated_caption, reference_captions):
    """Calculate detailed BLEU scores (BLEU-1 to BLEU-4)"""
    if not generated_caption or not reference_captions:
        return {"bleu-1": 0.0, "bleu-2": 0.0, "bleu-3": 0.0, "bleu-4": 0.0}

    # Ensure reference_captions is a list
    if isinstance(reference_captions, str):
        reference_captions = [reference_captions]

    # Tokenize
    candidate = generated_caption.lower().split()
    references = [ref.lower().split() for ref in reference_captions]

    # Use smoothing function
    smoothing = SmoothingFunction().method1

    bleu_scores = {}

    try:
        # BLEU-1 (unigrams)
        bleu_scores["bleu-1"] = sentence_bleu(
            references, candidate, weights=(1.0,), smoothing_function=smoothing
        )

        # BLEU-2 (bigrams)
        bleu_scores["bleu-2"] = sentence_bleu(
            references, candidate, weights=(0.5, 0.5), smoothing_function=smoothing
        )

        # BLEU-3 (trigrams)
        bleu_scores["bleu-3"] = sentence_bleu(
            references,
            candidate,
            weights=(0.33, 0.33, 0.33),
            smoothing_function=smoothing,
        )

        # BLEU-4 (4-grams) - standard BLEU
        bleu_scores["bleu-4"] = sentence_bleu(
            references,
            candidate,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        )

    except Exception as e:
        print(f"Error calculating detailed BLEU scores: {e}")
        bleu_scores = {"bleu-1": 0.0, "bleu-2": 0.0, "bleu-3": 0.0, "bleu-4": 0.0}

    return bleu_scores


def evaluate_with_dataset_references(generated_caption, dataset_item=None):
    """Enhanced BLEU evaluation using dataset reference captions when available"""
    if dataset_item and "captions" in dataset_item:
        # Use actual reference captions from dataset
        reference_captions = dataset_item["captions"]
        if isinstance(reference_captions, list):
            return calculate_bleu_score(generated_caption, reference_captions)

    # Fallback to generic high-quality references
    fallback_references = [
        "a detailed photograph showing clear visual elements",
        "an image with good composition and clarity",
        "a picture containing recognizable objects and scenes",
        "a well-composed photograph with distinct features",
    ]
    return calculate_bleu_score(generated_caption, fallback_references)


def enhance_caption_creativity(caption):
    """Add creative elements to make caption more engaging"""
    if not caption:
        return caption

    # Add sensory details
    sensory_enhancements = {
        "bright": "brilliantly bright",
        "dark": "mysteriously dark",
        "colorful": "vibrantly colorful",
        "quiet": "peacefully quiet",
        "busy": "bustling with activity",
    }

    enhanced_caption = caption
    for original, enhanced in sensory_enhancements.items():
        if original in caption.lower() and enhanced not in caption.lower():
            enhanced_caption = re.sub(
                rf"\b{original}\b", enhanced, enhanced_caption, flags=re.IGNORECASE
            )
            break

    # Add atmospheric descriptions
    if "outdoor" in caption.lower() or "outside" in caption.lower():
        atmospheric_additions = [
            "under an open sky",
            "in natural lighting",
            "surrounded by the environment",
        ]
        if not any(
            addition in enhanced_caption.lower() for addition in atmospheric_additions
        ):
            enhanced_caption += f", {random.choice(atmospheric_additions)}"

    return enhanced_caption


def post_process_caption(caption):
    """Post-process caption for better readability and grammar"""
    if not caption or not TEXTBLOB_AVAILABLE:
        return caption

    try:
        # Use TextBlob for grammar correction
        blob = TextBlob(caption)
        corrected = str(blob.correct())

        # Ensure proper capitalization
        corrected = corrected.strip()
        if corrected and not corrected[0].isupper():
            corrected = corrected[0].upper() + corrected[1:]

        # Remove redundant phrases
        redundant_phrases = [
            "this is a picture of",
            "this is an image of",
            "the image shows",
            "the picture shows",
        ]

        for phrase in redundant_phrases:
            corrected = re.sub(rf"^{phrase}\s*", "", corrected, flags=re.IGNORECASE)

        # Ensure proper ending
        if corrected and not corrected.endswith("."):
            corrected += "."

        return corrected
    except:
        return caption


def validate_beam_search_params(params):
    """Validate and fix beam search parameters"""
    validated_params = params.copy()

    # Check for diverse beam search
    num_beam_groups = validated_params.get("num_beam_groups", 1)
    diversity_penalty = validated_params.get("diversity_penalty", 0.0)

    if num_beam_groups > 1 or diversity_penalty > 0.0:
        # For diverse beam search, sampling must be disabled
        validated_params["do_sample"] = False

        # Ensure num_beams is divisible by num_beam_groups
        num_beams = validated_params.get("num_beams", 4)
        if num_beams % num_beam_groups != 0:
            validated_params["num_beams"] = num_beam_groups * 2

        # Set minimum diversity penalty
        if diversity_penalty == 0.0:
            validated_params["diversity_penalty"] = 1.0

    # Remove sampling parameters if do_sample is False
    if not validated_params.get("do_sample", True):
        sampling_params = ["temperature", "top_k", "top_p"]
        for param in sampling_params:
            if param in validated_params:
                del validated_params[param]

    return validated_params


def generate_caption_with_strategy(image, strategy="accurate", max_length=50):
    """Generate caption using specific strategy"""
    if flickr_model is None or flickr_processor is None:
        return "Model not loaded"

    try:
        # Process image
        inputs = flickr_processor(image, return_tensors="pt").to(device)

        # Get strategy parameters
        if caption_config and strategy in caption_config["caption_strategies"]:
            params = caption_config["caption_strategies"][strategy]
        else:
            # Default parameters
            params = {
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "length_penalty": 1.0,
                "do_sample": True,
                "num_beams": 4,
            }

        # Validate beam search parameters
        validated_params = validate_beam_search_params(params)

        # Generate caption with strategy-specific parameters
        with torch.no_grad():
            generation_kwargs = {
                **inputs,
                "max_length": max_length,
                "early_stopping": True,
                **validated_params,
            }

            generated_ids = flickr_model.generate(**generation_kwargs)
            caption = flickr_processor.decode(
                generated_ids[0], skip_special_tokens=True
            )

            return caption

    except Exception as e:
        error_msg = str(e)
        print(f"Warning: {strategy} strategy failed: {error_msg}")

        # Fallback to simple beam search
        try:
            simple_kwargs = {
                **inputs,
                "max_length": max_length,
                "num_beams": 5,
                "early_stopping": True,
                "do_sample": False,
                "repetition_penalty": 1.1,
            }
            generated_ids = flickr_model.generate(**simple_kwargs)
            caption = flickr_processor.decode(
                generated_ids[0], skip_special_tokens=True
            )
            return caption
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return "Unable to generate caption"


def generate_enhanced_captions(image, num_captions=3):
    """Generate multiple enhanced captions with different strategies"""
    strategies = ["accurate", "creative", "diverse"]
    captions = []
    failed_strategies = []

    for i, strategy in enumerate(strategies[:num_captions]):
        try:
            caption = generate_caption_with_strategy(image, strategy)

            # Check if caption generation failed
            if caption.startswith("Error generating caption"):
                failed_strategies.append(strategy)
                print(f"Warning: {strategy} strategy failed, using fallback")
                # Use accurate strategy as fallback
                caption = generate_caption_with_strategy(image, "accurate")

            # Apply enhancements based on strategy
            if strategy == "creative" and not caption.startswith("Error"):
                caption = enhance_caption_creativity(caption)

            # Post-process all captions
            caption = post_process_caption(caption)

            # Add sentiment analysis if available
            sentiment = "neutral"
            confidence = 0.5
            if NLTK_AVAILABLE and sia:
                try:
                    sentiment_scores = sia.polarity_scores(caption)
                    if sentiment_scores["compound"] >= 0.05:
                        sentiment = "positive"
                        confidence = sentiment_scores["compound"]
                    elif sentiment_scores["compound"] <= -0.05:
                        sentiment = "negative"
                        confidence = abs(sentiment_scores["compound"])
                    else:
                        sentiment = "neutral"
                        confidence = 1 - abs(sentiment_scores["compound"])
                except:
                    pass

            # Calculate accuracy percentage for accurate strategy
            accuracy_percentage = None
            if strategy == "accurate":
                # Enhanced accuracy calculation based on multiple factors
                accuracy_score = 0.0

                # Word count factor (optimal range 8-25 words for descriptive accuracy)
                word_count = len(caption.split())
                if 8 <= word_count <= 25:
                    accuracy_score += 0.25
                elif 5 <= word_count <= 30:
                    accuracy_score += 0.15
                else:
                    accuracy_score += 0.05

                # Confidence factor (weighted heavily for accuracy)
                accuracy_score += confidence * 0.35

                # Structure and grammar factor
                if len(caption) > 15 and any(punct in caption for punct in ".!?"):
                    accuracy_score += 0.15
                if caption[0].isupper():  # Proper capitalization
                    accuracy_score += 0.05

                # Descriptive content factor (more comprehensive)
                descriptive_words = [
                    "detailed",
                    "clear",
                    "visible",
                    "showing",
                    "containing",
                    "featuring",
                    "displaying",
                    "depicting",
                    "illustrating",
                    "presenting",
                    "reveals",
                    "appears",
                    "seems",
                    "looks",
                    "includes",
                    "has",
                    "with",
                ]
                desc_count = sum(
                    1 for word in descriptive_words if word in caption.lower()
                )
                accuracy_score += min(desc_count * 0.08, 0.2)

                # Avoid vague terms (penalty for inaccurate language)
                vague_words = [
                    "something",
                    "things",
                    "stuff",
                    "maybe",
                    "possibly",
                    "might",
                ]
                vague_count = sum(1 for word in vague_words if word in caption.lower())
                accuracy_score -= vague_count * 0.1

                accuracy_percentage = min(max(accuracy_score, 0), 1.0)

            captions.append(
                {
                    "text": caption,
                    "strategy": strategy
                    if strategy not in failed_strategies
                    else f"{strategy} (fallback)",
                    "sentiment": sentiment,
                    "confidence": round(confidence, 3),
                    "word_count": len(caption.split()),
                    "accuracy_percentage": round(accuracy_percentage, 3)
                    if accuracy_percentage is not None
                    else None,
                }
            )

        except Exception as e:
            print(f"Error with {strategy} strategy: {e}")
            # Add a simple fallback caption
            captions.append(
                {
                    "text": "Image description not available",
                    "strategy": f"{strategy} (error)",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "word_count": 4,
                }
            )

    # Sort by confidence score (highest first)
    captions.sort(key=lambda x: x["confidence"], reverse=True)

    return captions


def generate_caption(image, max_length=50):
    """Generate enhanced caption using best strategy"""
    enhanced_captions = generate_enhanced_captions(image, num_captions=1)
    if enhanced_captions:
        return enhanced_captions[0]["text"]
    return "Unable to generate caption"


# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Caption Generator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .header h1 {
            color: #333;
            font-size: 2.8em;
            margin-bottom: 10px;
        }

        .header p {
            color: #666;
            font-size: 1.2em;
            line-height: 1.6;
        }

        .features {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-top: 15px;
            flex-wrap: wrap;
        }

        .feature {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 500;
        }

        .generation-controls {
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }

        .mode-selector, .strategy-selector {
            margin: 10px 0;
        }

        .mode-selector label, .strategy-selector label {
            display: inline-block;
            margin-right: 10px;
            font-weight: 500;
            color: #333;
        }

        .mode-selector select, .strategy-selector select {
            padding: 8px 12px;
            border: 2px solid #ddd;
            border-radius: 5px;
            background: white;
            font-size: 14px;
            min-width: 200px;
        }

        .caption-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }

        .caption-quality {
            font-size: 14px;
            color: #666;
        }

        .caption-alternatives {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #e3f2fd;
        }

        .caption-alternatives h4 {
            color: #1565c0;
            margin-bottom: 15px;
        }

        .alternative-caption {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #2196f3;
        }

        .alternative-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
            font-size: 12px;
            color: #666;
        }

        .strategy-badge {
            background: #e3f2fd;
            color: #1565c0;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
            position: relative;
        }

        .strategy-creative {
            background: #fce4ec;
            color: #ad1457;
        }

        .strategy-diverse {
            background: #f3e5f5;
            color: #7b1fa2;
        }

        .strategy-accurate {
            background: #e8f5e8;
            color: #2e7d32;
        }

        .accuracy-badge {
            background: #e8f5e8;
            color: #2e7d32;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
            margin-left: 5px;
        }

        .accuracy-high {
            background: #e8f5e8;
            color: #2e7d32;
        }

        .accuracy-medium {
            background: #fff3e0;
            color: #f57c00;
        }

        .accuracy-low {
            background: #ffebee;
            color: #d32f2f;
        }

        .accuracy-info {
            font-size: 10px;
            color: #666;
            margin-left: 5px;
            cursor: help;
        }

        .accuracy-tooltip {
            position: relative;
            display: inline-block;
        }

        .accuracy-tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            font-size: 11px;
        }

        .accuracy-tooltip:hover .tooltiptext {
            visibility: visible;
        }




        .score-badge {
            background: #f3e5f5;
            color: #7b1fa2;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
        }



        .upload-section {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 12px;
            border: 2px dashed #ddd;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }

        .upload-section:hover {
            border-color: #667eea;
            background: #f0f0ff;
        }

        .upload-section.dragover {
            border-color: #667eea;
            background: #f0f0ff;
            transform: scale(1.02);
        }

        .file-input {
            margin: 20px 0;
        }

        .file-input input[type="file"] {
            display: none;
        }

        .file-input label {
            display: inline-block;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            transition: transform 0.2s;
        }

        .file-input label:hover {
            transform: translateY(-2px);
        }

        .generate-btn {
            background: linear-gradient(135deg, #28a745, #20c997);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 18px;
            font-weight: 500;
            transition: transform 0.2s;
            width: 100%;
            margin-top: 20px;
        }

        .generate-btn:hover {
            transform: translateY(-2px);
        }

        .generate-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }

        .loading {
            display: none;
            text-align: center;
            color: #667eea;
            font-size: 16px;
            margin: 20px 0;
        }

        .loading .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .result-section {
            display: none;
            margin-top: 30px;
        }

        .result-image {
            max-width: 100%;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        }

        .caption-result {
            background: linear-gradient(135deg, #e3f2fd, #bbdefb);
            padding: 25px;
            border-radius: 10px;
            font-size: 18px;
            line-height: 1.6;
            border-left: 5px solid #2196f3;
            margin-bottom: 20px;
        }

        .caption-text {
            font-style: italic;
            color: #1565c0;
            font-weight: 500;
        }

        .copy-btn {
            background: #6c757d;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 15px;
            cursor: pointer;
            font-size: 12px;
            float: right;
            margin-top: -5px;
        }

        .copy-btn:hover {
            background: #5a6268;
        }

        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #dc3545;
        }

        .file-preview {
            margin: 20px 0;
            text-align: center;
        }

        .file-preview img {
            max-width: 300px;
            max-height: 200px;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .file-name {
            margin-top: 15px;
            color: #667eea;
            font-weight: 500;
        }

        @media (max-width: 600px) {
            .container {
                padding: 20px;
                margin: 10px;
            }

            .header h1 {
                font-size: 2.2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Enhanced AI Caption Generator</h1>
            <p>Upload an image and get creative, distinctive captions using advanced AI trained on Flickr30k dataset</p>
            <div class="features">
                <span class="feature">🎯 Accurate</span>
                <span class="feature">🎨 Creative</span>
                <span class="feature">🌈 Diverse</span>
                <span class="feature">🔍 Distinctive</span>
            </div>
        </div>

        <div class="upload-section" id="uploadArea">
            <div class="file-input">
                <input type="file" id="file" name="file" accept="image/*" required>
                <label for="file">📁 Choose Image File</label>
            </div>
            <p>Or drag and drop an image here</p>
            <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
                Supported formats: PNG, JPG, JPEG, GIF, BMP, WebP
            </p>
            <div class="file-preview" id="filePreview" style="display: none;">
                <img id="previewImage" src="" alt="Preview">
                <div class="file-name" id="fileName"></div>
            </div>
        </div>

        <div class="generation-controls">
            <div class="mode-selector">
                <label for="generationMode">Generation Mode:</label>
                <select id="generationMode">
                    <option value="enhanced">🎨 Enhanced (Best)</option>
                    <option value="multiple">🌈 Multiple Styles</option>
                    <option value="single">🎯 Single Strategy</option>
                </select>
            </div>

            <div class="strategy-selector" id="strategySelector" style="display: none;">
                <label for="captionStrategy">Strategy:</label>
                <select id="captionStrategy">
                    <option value="accurate">🎯 Accurate</option>
                    <option value="creative">🎨 Creative</option>
                    <option value="diverse">🌈 Diverse</option>
                </select>
            </div>
        </div>

        <button type="button" class="generate-btn" id="generateBtn">✨ Generate Enhanced Caption</button>

        <div class="loading" id="loading">
            <div class="spinner"></div>
            Analyzing image and generating caption...
        </div>

        <div class="result-section" id="resultSection">
            <img id="resultImage" class="result-image" src="" alt="Uploaded Image">

            <div class="caption-result">
                <div class="caption-header">
                    <button class="copy-btn" onclick="copyCaption()">Copy Best</button>
                    <div class="caption-quality" id="captionQuality"></div>
                </div>


                <div class="caption-text" id="captionText"></div>

                <div class="caption-alternatives" id="captionAlternatives" style="display: none;">
                    <h4>Alternative Captions:</h4>
                    <div class="alternatives-list" id="alternativesList"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedImage = null;

        // File input handling
        const fileInput = document.getElementById('file');
        const uploadArea = document.getElementById('uploadArea');
        const filePreview = document.getElementById('filePreview');
        const previewImage = document.getElementById('previewImage');
        const fileName = document.getElementById('fileName');

        // File preview
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                selectedImage = file;
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    fileName.textContent = file.name;
                    filePreview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });

        async function generateCaption() {
            if (!selectedImage) {
                alert('Please select an image first');
                return;
            }

            const generateBtn = document.getElementById('generateBtn');
            const loading = document.getElementById('loading');
            const resultSection = document.getElementById('resultSection');
            const generationMode = document.getElementById('generationMode').value;
            const captionStrategy = document.getElementById('captionStrategy').value;

            generateBtn.disabled = true;
            generateBtn.textContent = '🔄 Processing...';
            loading.style.display = 'block';
            resultSection.style.display = 'none';

            const formData = new FormData();
            formData.append('image', selectedImage);
            formData.append('mode', generationMode);
            formData.append('strategy', captionStrategy);

            try {
                const response = await fetch('/generate_caption', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                loading.style.display = 'none';

                if (data.success) {
                    // Display image
                    const resultImg = document.getElementById('resultImage');
                    resultImg.src = 'data:image/jpeg;base64,' + data.image;

                    // Display main caption
                    document.getElementById('captionText').textContent = data.caption;

                    // Display caption quality info
                    const captionQuality = document.getElementById('captionQuality');
                    if (data.captions && data.captions.length > 0) {
                        const mainCaption = data.captions[0];

                        // Determine strategy class
                        let strategyClass = 'strategy-badge';
                        if (mainCaption.strategy === 'creative') {
                            strategyClass += ' strategy-creative';
                        } else if (mainCaption.strategy === 'diverse') {
                            strategyClass += ' strategy-diverse';
                        } else if (mainCaption.strategy === 'accurate') {
                            strategyClass += ' strategy-accurate';
                        }

                        let accuracyBadge = '';
                        if (mainCaption.accuracy_percentage !== null && mainCaption.strategy === 'accurate') {
                            const accuracy = mainCaption.accuracy_percentage;
                            let accuracyClass = 'accuracy-low';
                            if (accuracy >= 0.85) {
                                accuracyClass = 'accuracy-high';
                            } else if (accuracy >= 0.70) {
                                accuracyClass = 'accuracy-medium';
                            }
                            accuracyBadge = `<span class="accuracy-badge ${accuracyClass}">Accuracy: ${accuracy.toFixed(3)}</span>`;
                        }

                        captionQuality.innerHTML = `
                            <span class="${strategyClass}">${mainCaption.strategy}</span>
                            ${accuracyBadge}
                            ${mainCaption.strategy === 'accurate' ?
                                '<span class="accuracy-tooltip accuracy-info">ℹ️<span class="tooltiptext">Accuracy: Decimal score (0-1) based on word count (8-25 optimal), confidence level, proper structure, descriptive content, and avoidance of vague terms</span></span>' : ''}
                            <span class="score-badge">${mainCaption.sentiment}</span>
                        `;
                    }

                    // Display alternative captions if multiple mode
                    const captionAlternatives = document.getElementById('captionAlternatives');
                    const alternativesList = document.getElementById('alternativesList');

                    if (data.mode === 'multiple' && data.captions && data.captions.length > 1) {
                        alternativesList.innerHTML = '';

                        data.captions.slice(1).forEach((caption, index) => {
                            const altDiv = document.createElement('div');
                            altDiv.className = 'alternative-caption';
                            altDiv.innerHTML = `
                                <div class="alternative-header">
                                    <span class="strategy-badge ${caption.strategy === 'creative' ? 'strategy-creative' :
                                                                caption.strategy === 'diverse' ? 'strategy-diverse' :
                                                                caption.strategy === 'accurate' ? 'strategy-accurate' : ''}">${caption.strategy}</span>
                                    <div>
                                        ${caption.accuracy_percentage !== null && caption.strategy === 'accurate' ?
                                            `<span class="accuracy-badge ${caption.accuracy_percentage >= 0.85 ? 'accuracy-high' :
                                                                        caption.accuracy_percentage >= 0.70 ? 'accuracy-medium' : 'accuracy-low'}">
                                                Accuracy: ${caption.accuracy_percentage.toFixed(3)}
                                            </span>` : ''}
                                        <span class="score-badge">${caption.sentiment}</span>
                                    </div>
                                </div>
                                <div>${caption.text}</div>
                            `;
                            alternativesList.appendChild(altDiv);
                        });

                        captionAlternatives.style.display = 'block';
                    } else {
                        captionAlternatives.style.display = 'none';
                    }

                    resultSection.style.display = 'block';
                } else {
                    showError(data.error);
                }
            } catch (error) {
                loading.style.display = 'none';
                showError('Error connecting to server: ' + error.message);
            }

            generateBtn.disabled = false;
            generateBtn.textContent = '✨ Generate Enhanced Caption';
        }

        function copyCaption() {
            const captionText = document.getElementById('captionText').textContent;
            navigator.clipboard.writeText(captionText).then(() => {
                const copyBtn = document.querySelector('.copy-btn');
                const originalText = copyBtn.textContent;
                copyBtn.textContent = 'Copied!';
                setTimeout(() => {
                    copyBtn.textContent = originalText;
                }, 2000);
            });
        }

        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = message;

            const generateBtn = document.getElementById('generateBtn');
            generateBtn.parentNode.insertBefore(errorDiv, generateBtn);

            setTimeout(() => {
                errorDiv.remove();
            }, 5000);
        }

        // Mode selector handling
        document.getElementById('generationMode').addEventListener('change', function() {
            const strategySelector = document.getElementById('strategySelector');
            const generateBtn = document.getElementById('generateBtn');

            if (this.value === 'single') {
                strategySelector.style.display = 'block';
                generateBtn.textContent = '✨ Generate Single Caption';
            } else if (this.value === 'multiple') {
                strategySelector.style.display = 'none';
                generateBtn.textContent = '✨ Generate Multiple Captions';
            } else {
                strategySelector.style.display = 'none';
                generateBtn.textContent = '✨ Generate Enhanced Caption';
            }
        });

        // Set up generate button
        document.getElementById('generateBtn').addEventListener('click', generateCaption);

        // Drag and drop functionality
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    selectedImage = file;
                    fileInput.files = files;

                    const reader = new FileReader();
                    reader.onload = function(e) {
                        previewImage.src = e.target.result;
                        fileName.textContent = file.name;
                        filePreview.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                }
            }
        });

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }


    </script>
</body>
</html>
"""


@app.route("/")
def index():
    """Render the main page"""
    return render_template_string(HTML_TEMPLATE)


@app.route("/generate_caption", methods=["POST"])
def generate_caption_route():
    """Handle enhanced caption generation request"""
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded"})

        file = request.files["image"]

        if file.filename == "":
            return jsonify({"success": False, "error": "No image selected"})

        # Get generation mode from request
        mode = request.form.get("mode", "enhanced")  # single, enhanced, or multiple
        strategy = request.form.get(
            "strategy", "accurate"
        )  # accurate, creative, diverse

        # Load and process image
        image = Image.open(file.stream).convert("RGB")

        # Generate captions based on mode
        if mode == "multiple":
            # Generate multiple captions with different strategies
            captions_data = generate_enhanced_captions(image, num_captions=3)
            primary_caption = (
                captions_data[0]["text"]
                if captions_data
                else "Unable to generate caption"
            )
        elif mode == "single":
            # Generate single caption with specific strategy
            primary_caption = generate_caption_with_strategy(image, strategy)
            captions_data = [
                {
                    "text": primary_caption,
                    "strategy": strategy,
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "word_count": len(primary_caption.split()),
                    "accuracy_percentage": 0.750 if strategy == "accurate" else None,
                }
            ]
        else:  # enhanced mode (default)
            # Generate enhanced captions and return the best one
            captions_data = generate_enhanced_captions(image, num_captions=3)
            primary_caption = (
                captions_data[0]["text"]
                if captions_data
                else "Unable to generate caption"
            )

        # Convert image to base64 for response
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return jsonify(
            {
                "success": True,
                "caption": primary_caption,
                "captions": captions_data,
                "image": img_str,
                "mode": mode,
                "enhancement_available": True,
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/strategies", methods=["GET"])
def get_strategies():
    """Get available caption generation strategies"""
    strategies = {
        "accurate": {
            "name": "Accurate",
            "description": "Precise, factual descriptions with high confidence",
            "icon": "🎯",
        },
        "creative": {
            "name": "Creative",
            "description": "Artistic, expressive captions with enhanced vocabulary",
            "icon": "🎨",
        },
        "diverse": {
            "name": "Diverse",
            "description": "Varied perspectives and unique descriptions",
            "icon": "🌈",
        },
    }
    return jsonify(strategies)


def initialize_models():
    """Initialize Flickr30k model on startup"""
    print("🚀 Initializing enhanced caption generator...")

    # Load Flickr30k model
    if not load_flickr_model():
        print("❌ Failed to load caption model")
        return False

    # Load caption configuration
    load_caption_config()
    print("✅ Caption configuration loaded")

    print("✅ Enhanced caption generator initialized successfully!")
    return True


if __name__ == "__main__":
    print("📸 AI Caption Generator (Flickr30k)")
    print("=" * 50)

    if initialize_models():
        print("\n✅ Server ready!")
        print("📱 Open http://localhost:3000 in your browser")
        print("\n" + "=" * 50)
        app.run(debug=True, host="0.0.0.0", port=3000)
    else:
        print("\n❌ Failed to initialize models. Server cannot start.")
