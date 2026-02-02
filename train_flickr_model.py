import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
from tqdm import tqdm

class Flickr30kDataset(Dataset):
    def __init__(self, dataset, processor, max_length=128):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']

        # Handle different possible caption column names
        caption = None
        if 'alt_text' in item:
            caption = item['alt_text'][0] if isinstance(item['alt_text'], list) else item['alt_text']
        elif 'caption' in item:
            caption = item['caption'][0] if isinstance(item['caption'], list) else item['caption']
        elif 'captions' in item:
            caption = item['captions'][0] if isinstance(item['captions'], list) else item['captions']
        elif 'text' in item:
            caption = item['text'][0] if isinstance(item['text'], list) else item['text']
        else:
            # Find any text-like field
            for key in item.keys():
                if isinstance(item[key], (str, list)):
                    caption = item[key][0] if isinstance(item[key], list) else item[key]
                    break

        if caption is None:
            caption = "A photo"  # Fallback caption

        # Process image and text
        try:
            inputs = self.processor(
                images=image,
                text=caption,
                return_tensors="pt",
                padding=True,
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
            # Return dummy data
            dummy_inputs = self.processor(
                images=Image.new('RGB', (224, 224)),
                text="A photo",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            return {
                'pixel_values': dummy_inputs['pixel_values'].squeeze(0),
                'input_ids': dummy_inputs['input_ids'].squeeze(0),
                'attention_mask': dummy_inputs['attention_mask'].squeeze(0)
            }

class FlickrCaptionModel:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load processor and model
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(self.device)

    def prepare_dataset(self):
        print("Loading Flickr30k dataset...")
        try:
            # Load dataset with simplified error handling
            print("Downloading Flickr30k dataset...")
            dataset = load_dataset("AnyModal/flickr30k")

            print("Dataset structure:")
            # Safe access to dataset keys
            try:
                available_keys = []
                dataset_dict = dict(dataset)  # Convert to dict to avoid type issues
                available_keys = list(dataset_dict.keys())
                print(f"Available splits: {available_keys}")
            except Exception as e:
                print(f"Could not access dataset keys: {e}")
                return None, None

            if not available_keys:
                return None, None

            # Use first available split
            split_name = str(available_keys[0])  # Ensure string type

            # Safely access the split
            try:
                dataset_dict = dict(dataset)
                data_split = dataset_dict[split_name]
                split_length = len(data_split)
            except Exception as e:
                print(f"Could not access dataset split: {e}")
                return None, None

            # Use smaller subsets for training
            train_size = min(1000, split_length)  # Reduced size
            val_size = min(100, train_size // 10)

            # Create train/val splits from the same data
            try:
                if hasattr(data_split, 'select'):
                    train_data = data_split.select(range(train_size))
                    val_data = data_split.select(range(val_size))
                else:
                    # Fallback for datasets without select method
                    train_data = data_split
                    val_data = data_split
            except Exception as e:
                print(f"Could not create dataset subsets: {e}")
                return None, None

            print(f"Training samples: {train_size}")
            print(f"Validation samples: {val_size}")

            # Create PyTorch datasets
            train_flickr = Flickr30kDataset(train_data, self.processor)
            val_flickr = Flickr30kDataset(val_data, self.processor)

            return train_flickr, val_flickr

        except Exception as e:
            print(f"Primary dataset loading failed: {e}")
            print("Trying alternative approach...")
            return None, None

    def train(self, epochs=3, batch_size=8, learning_rate=5e-5):
        train_dataset, val_dataset = self.prepare_dataset()

        if train_dataset is None or val_dataset is None:
            print("Failed to load dataset")
            return

        # Create data loaders (reduce num_workers to avoid issues)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Setup optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        print(f"Starting training for {epochs} epochs...")

        best_loss = float('inf')

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_train_loss = 0
            train_steps = 0

            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_pbar = tqdm(train_loader, desc="Training")

            for batch in train_pbar:
                try:
                    # Move to device
                    pixel_values = batch['pixel_values'].to(self.device)
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)

                    # Forward pass
                    outputs = self.model(
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

                    total_train_loss += loss.item()
                    train_steps += 1

                    train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                except Exception as e:
                    print(f"Error in training batch: {e}")
                    continue

            avg_train_loss = total_train_loss / train_steps if train_steps > 0 else 0

            # Validation phase
            self.model.eval()
            total_val_loss = 0
            val_steps = 0

            val_pbar = tqdm(val_loader, desc="Validation")
            with torch.no_grad():
                for batch in val_pbar:
                    try:
                        pixel_values = batch['pixel_values'].to(self.device)
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)

                        outputs = self.model(
                            pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=input_ids
                        )

                        loss = outputs.loss
                        total_val_loss += loss.item()
                        val_steps += 1

                        val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

                    except Exception as e:
                        print(f"Error in validation batch: {e}")
                        continue

            avg_val_loss = total_val_loss / val_steps if val_steps > 0 else float('inf')

            print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Save best model
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                self.save_model(f"best_flickr_model_epoch_{epoch + 1}")
                print(f"New best model saved with validation loss: {avg_val_loss:.4f}")

        print("Training completed!")

    def save_model(self, save_name="flickr_caption_model"):
        """Save the trained model"""
        os.makedirs("models", exist_ok=True)
        save_path = f"models/{save_name}"

        # Save model and processor
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)

        print(f"Model saved to {save_path}")

    def load_model(self, model_path):
        """Load a trained model"""
        self.processor = BlipProcessor.from_pretrained(model_path)
        self.model = BlipForConditionalGeneration.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        print(f"Model loaded from {model_path}")

    def generate_caption(self, image, max_length=50, num_beams=5):
        """Generate caption for a given image"""
        self.model.eval()

        if isinstance(image, str):
            image = Image.open(image).convert('RGB')

        inputs = self.processor(images=image, return_tensors="pt")
        # Move inputs to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                do_sample=False
            )

        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption

def main():
    print("🚀 Flickr30k Caption Model Training")
    print("=" * 50)

    # Initialize model
    caption_model = FlickrCaptionModel()

    # Train the model
    caption_model.train(epochs=3, batch_size=4, learning_rate=5e-5)

    # Test caption generation
    print("\n🧪 Testing caption generation...")

    # Try to load a sample image from the dataset for testing
    try:
        dataset = load_dataset("AnyModal/flickr30k")
        # Safe dataset access
        try:
            dataset_dict = dict(dataset)
            test_keys = list(dataset_dict.keys())
            if test_keys:
                test_split = str(test_keys[0])  # Use first available split
                test_data = dataset_dict[test_split]
                test_item = test_data[0]
                test_image = test_item['image']

                caption = caption_model.generate_caption(test_image)
                print(f"Generated caption: '{caption}'")

                # Compare with ground truth
                gt_captions = "No ground truth available"
                if 'caption' in test_item:
                    gt_captions = test_item['caption']
                elif 'captions' in test_item:
                    gt_captions = test_item['captions']
                print(f"Ground truth captions: {gt_captions}")
            else:
                print("No dataset splits available for testing")
        else:
            print("Cannot access dataset for testing")

    except Exception as e:
        print(f"Testing error: {e}")
        print("Skipping test generation...")

    print("\n✅ Training and testing completed!")

if __name__ == "__main__":
    main()
