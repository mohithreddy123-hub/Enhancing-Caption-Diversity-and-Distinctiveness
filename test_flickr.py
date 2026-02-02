#!/usr/bin/env python3
"""
Quick test script for Flickr30k caption generation without full training
This uses the base BLIP model to generate captions similar to Flickr30k style
"""

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

def test_flickr_captions():
    print("📸 Testing Flickr-style Caption Generation")
    print("=" * 50)

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Load base BLIP model (similar to what would be fine-tuned on Flickr30k)
        print("\n🔄 Loading BLIP model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model = model.to(device)
        model.eval()
        print("✅ Model loaded successfully!")

        # Test with sample images if available
        test_images = []

        # Check for sample images in data folder
        if os.path.exists('data/train2017'):
            import glob
            sample_files = glob.glob('data/train2017/*.jpg')[:3]
            test_images.extend(sample_files)

        # If no sample images, create a simple test
        if not test_images:
            print("ℹ️  No sample images found in data folder")
            print("   Place some .jpg images in data/train2017/ for testing")
            return

        print(f"\n🧪 Testing with {len(test_images)} sample images...")

        for i, img_path in enumerate(test_images, 1):
            try:
                # Load and process image
                image = Image.open(img_path).convert('RGB')

                # Generate caption
                inputs = processor(image, return_tensors="pt")
                # Move inputs to device
                inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_length=50,
                        num_beams=5,
                        early_stopping=True,
                        do_sample=False
                    )

                caption = processor.decode(generated_ids[0], skip_special_tokens=True)

                print(f"\n📷 Image {i}: {os.path.basename(img_path)}")
                print(f"📝 Caption: \"{caption}\"")

            except Exception as e:
                print(f"❌ Error processing image {i}: {e}")

        print("\n✅ Caption generation test completed!")
        print("🎯 These are baseline captions. For better results:")
        print("   1. Run: python train_flickr_model.py")
        print("   2. This will fine-tune on Flickr30k dataset")
        print("   3. Then use app_flickr.py for improved captions")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        print("Make sure transformers is installed: pip install transformers")

def generate_sample_caption(image_path):
    """Generate a single caption for testing"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model = model.to(device)
        model.eval()

        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        # Move inputs to device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )

        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    test_flickr_captions()
