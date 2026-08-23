# this is to test one imager to make sure the model and flow works


# Import Path so we can work with folders and file paths.
from pathlib import Path

# Import NumPy to inspect and normalise the embedding.
import numpy as np

# Import PyTorch because CLIP uses PyTorch.
import torch

# Import Image from Pillow so we can open the clothing image.
from PIL import Image

# Import the pre-trained CLIP model and image processor.
from transformers import CLIPModel, CLIPProcessor


# Name of the pre-trained CLIP model.
MODEL_NAME = "openai/clip-vit-base-patch32"

# Folder containing the cleaned clothing images.
CLEAN_IMAGE_FOLDER = Path("Data/processed/clean_images")


def get_device():
    """
    Select the best processing device available on the computer.
    """

    # Use an NVIDIA GPU if one is available.
    if torch.cuda.is_available():
        return "cuda"

    # Use Apple's Metal GPU if available.
    if torch.backends.mps.is_available():
        return "mps"

    # Use the CPU if no compatible GPU is available.
    return "cpu"


def find_first_image():
    """
    Find the first valid image in the cleaned image folder.
    """

    # These are the image file extensions we support.
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    # Check that the cleaned image folder exists.
    if not CLEAN_IMAGE_FOLDER.exists():
        raise FileNotFoundError(
            f"Clean image folder not found: {CLEAN_IMAGE_FOLDER}"
        )

    # Find supported image files in the cleaned image folder.
    image_files = [
        image_path
        for image_path in CLEAN_IMAGE_FOLDER.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in image_extensions
    ]

    # Stop if the folder contains no images.
    if not image_files:
        raise FileNotFoundError(
            f"No image files found in: {CLEAN_IMAGE_FOLDER}"
        )

    # Sort the files so the same image is selected every time.
    image_files.sort()

    # Return the first image.
    return image_files[0]


def create_one_embedding(image_path, model, processor, device):
    """
    Convert one clothing image into a CLIP embedding.
    """

    # Open the image and convert it to RGB.
    image = Image.open(image_path).convert("RGB")

    # Convert the image into tensors expected by CLIP.
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Move the input tensors to the selected device.
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Turn off gradient calculations because we are not training CLIP.
    with torch.no_grad():

        # Ask CLIP for image features.
        # return_dict=False makes the output easier to handle.
        image_output = model.get_image_features(
            **inputs,
            return_dict=False
        )

    # Support both a direct tensor and tuple-style output.
    if isinstance(image_output, tuple):

        # Use the first item if the model returns a tuple.
        embedding_tensor = image_output[0]

    else:

        # Use the result directly if it is already a tensor.
        embedding_tensor = image_output

   # Remove the batch dimension.
    embedding_tensor = embedding_tensor[0]


# This changes the shape from (50, 768) to (768,).
    embedding_tensor = embedding_tensor.mean(dim=0)

# Move the final image vector to the CPU and convert it to NumPy.
    embedding = embedding_tensor.cpu().numpy()

    # Calculate the length of the embedding vector.
    embedding_length = np.linalg.norm(embedding)

    # Stop if the embedding cannot be normalised.
    if embedding_length == 0:
        raise ValueError("The embedding has a length of zero.")

    # Normalise the embedding for cosine-similarity comparisons.
    embedding = embedding / embedding_length

    # Return the final NumPy embedding.
    return embedding


def test_one_image():
    """
    Test the complete process using one cleaned clothing image.
    """

    # Select the best available device.
    device = get_device()

    # Display the selected device.
    print(f"Using device: {device}")

    # Load the image processor.
    print("Loading CLIP processor...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # Load the pre-trained CLIP model.
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME)

    # Move the model to the selected device.
    model.to(device)

    # Put the model into evaluation mode.
    model.eval()

    # Find one cleaned image.
    image_path = find_first_image()

    # Create the embedding for that one image.
    embedding = create_one_embedding(
        image_path=image_path,
        model=model,
        processor=processor,
        device=device
    )

    # Check whether the embedding contains valid numerical values.
    contains_invalid_values = not np.isfinite(embedding).all()

    # Print the test results.
    print("\n--- One Image Embedding Test ---")
    print(f"Image tested: {image_path}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding data type: {embedding.dtype}")
    print(f"First five values: {embedding[:5]}")
    print(f"Contains invalid values: {contains_invalid_values}")

    # Confirm whether the test passed.
    if contains_invalid_values:
        print("Test failed: the embedding contains invalid values.")
    else:
        print("Test passed: one valid image embedding was created.")


# Run the test only when this file is executed directly.
if __name__ == "__main__":
    test_one_image()