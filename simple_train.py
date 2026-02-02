#!/usr/bin/env python3
"""
Simple and robust Flickr30k training script with better error handling
"""

import torch

import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SimpleFlickrDataset(Dataset):
    def __init__(self, images, captions, processor, max_length=50):
        self.images = images
        self.captions = captions
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = self.images[idx]
            caption = self.captions[idx]

            # Ensure image is PIL Image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert('RGB')

            # Process image and text
            inputs = self.processor(
                images=image,
                text=caption,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )

            return {
                'pixel_values': inputs['pixel_values'].squeeze(0),
                'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0)
            }
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return a dummy item
            dummy_image = Image.new('RGB', (224, 224), color='white')
            dummy_inputs = self.processor(
                images=dummy_image,
                text="A photo",
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=self.max_length
            )
            return {
                'pixel_values': dummy_inputs['pixel_values'].squeeze(0),
                'input_ids': dummy_inputs['input_ids'].squeeze(0),
                'attention_mask': dummy_inputs['attention_mask'].squeeze(0)
            }

def load_flickr_data():
    """Load Flickr30k dataset with robust error handling"""
    print("🔄 Loading Flickr30k dataset...")

    dataset = None
    try:
        # Try primary dataset
        dataset = load_dataset("AnyModal/flickr30k")
        print("✅ Loaded AnyModal/flickr30k")

    except Exception as e:
        print(f"❌ Failed to load AnyModal/flickr30k: {e}")
        try:
            # Try alternative dataset
            dataset = load_dataset("nlphuji/flickr30k")
            print("✅ Loaded nlphuji/flickr30k")
        except Exception as e2:
            print(f"❌ Failed to load alternative dataset: {e2}")
            return None, None, None, None

    if dataset is None:
        return None, None, None, None

    # Explore dataset structure
    try:
        # Get available splits using dict conversion to avoid type issues
        dataset_dict = dict(dataset)
        available_keys = list(dataset_dict.keys())

        if not available_keys:
            print("No dataset splits found")
            return None, None, None, None

        print(f"Dataset keys: {available_keys}")

        # Use the first available split
        split_name = str(available_keys[0])  # Ensure string type
        data_split = dataset_dict[split_name]

        # Check structure of first item
        sample_item = data_split[0]
        sample_keys = []
        try:
            if hasattr(sample_item, 'keys'):
                sample_keys = list(sample_item.keys())
            else:
                print("Sample item doesn't have keys method")
                return None, None, None, None
        except Exception:
            print("Cannot access sample item keys")
            return None, None, None, None
        print(f"Sample item keys: {sample_keys}")
    except Exception as e:
        print(f"Error exploring dataset structure: {e}")
        return None, None, None, None

    # Extract images and captions
    images = []
    captions = []

    # Determine caption field name
    caption_field = None
    for field in ['alt_text', 'caption', 'captions', 'text', 'sentence']:
        if field in sample_keys:
            caption_field = field
            break

    if caption_field is None:
        print("❌ No caption field found in dataset")
        return None, None, None, None

    print(f"Using caption field: '{caption_field}'")

    # Load data (use smaller subset for testing)
    data_length = 2000  # default fallback
    try:
        if hasattr(data_split, '__len__'):
            data_length = len(data_split)
        else:
            print("Dataset doesn't support len(), using default")
    except Exception:
        print("Cannot get dataset length, using default of 2000")

    max_samples = min(2000, data_length)
    print(f"Loading {max_samples} samples...")

    for i in tqdm(range(max_samples), desc="Loading data"):
        try:
            # Safe item access
            if hasattr(data_split, '__getitem__'):
                item = data_split[i]
            else:
                print("Dataset doesn't support indexing")
                break

            # Safely get image
            image = None
            if hasattr(item, '__contains__') and 'image' in item:
                image = item['image']

            # Safely get caption
            caption = None
            if caption_field and hasattr(item, '__contains__') and caption_field in item:
                caption_data = item[caption_field]
                if isinstance(caption_data, list) and len(caption_data) > 0:
                    caption = str(caption_data[0])
                elif caption_data:
                    caption = str(caption_data)

            # Validate data
            if image is not None and caption and len(caption.strip()) > 0:
                images.append(image)
                captions.append(caption.strip())

        except Exception as e:
            print(f"Skipping item {i}: {e}")
            continue

    print(f"Successfully loaded {len(images)} image-caption pairs")

    # Split into train/val
    split_idx = int(0.8 * len(images))
    train_images = images[:split_idx]
    train_captions = captions[:split_idx]
    val_images = images[split_idx:]
    val_captions = captions[split_idx:]

    return train_images, train_captions, val_images, val_captions

def train_model():
    print("📸 Simple Flickr30k Training Script")
    print("=" * 50)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_images, train_captions, val_images, val_captions = load_flickr_data()

    if train_images is None:
        print("❌ Failed to load dataset")
        return

    if train_images is None or val_images is None:
        print("❌ No training data available")
        return

    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")

    # Load model and processor
    print("\n🔄 Loading BLIP model...")
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        # Move model to device
        model = model.to(device)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Create datasets
    train_dataset = SimpleFlickrDataset(train_images, train_captions, processor)
    val_dataset = SimpleFlickrDataset(val_images, val_captions, processor)

    # Create data loaders (no multiprocessing to avoid errors)
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Setup training
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 2

    print(f"\n🚀 Starting training for {epochs} epochs...")

    best_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_batches = 0

        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_pbar = tqdm(train_loader, desc="Training")

        for batch in train_pbar:
            try:
                # Move to device
                pixel_values = batch['pixel_values'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # Forward pass
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )

                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            except Exception as e:
                print(f"Error in batch: {e}")
                continue

        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                try:
                    pixel_values = batch['pixel_values'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)

                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids
                    )

                    val_loss += outputs.loss.item()
                    val_batches += 1

                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue

        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        print(f"Average validation loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = f"models/flickr_model_epoch_{epoch + 1}"
            os.makedirs("models", exist_ok=True)

            try:
                model.save_pretrained(save_path)
                processor.save_pretrained(save_path)
                print(f"✅ Best model saved to {save_path}")
            except Exception as e:
                print(f"❌ Failed to save model: {e}")

    print("\n🎉 Training completed!")

    # Test generation
    print("\n🧪 Testing caption generation...")
    try:
        model.eval()
        if val_images and len(val_images) > 0 and val_captions and len(val_captions) > 0:
            test_image = val_images[0]

            inputs = processor(test_image, return_tensors="pt")
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
            print(f"Generated caption: '{caption}'")
            print(f"Ground truth: '{val_captions[0]}'")
        else:
            print("No validation data available for testing")

    except Exception as e:
        print(f"Testing error: {e}")

    print("\n✅ All done! Model saved in models/ directory")

if __name__ == "__main__":
    train_model()
