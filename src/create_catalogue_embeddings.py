# Import Path to work with image folders and file paths.
from pathlib import Path

# Import NumPy to store and save embedding arrays.
import numpy as np

# Import PyTorch to run the CLIP model.
import torch

# Import Pillow to open images.
from PIL import Image

# Import CLIP model and image processor.
from transformers import CLIPModel, CLIPProcessor

# Import tqdm to display progress.
from tqdm import tqdm


# Name of the pre-trained CLIP model.
MODEL_NAME = "openai/clip-vit-base-patch32"

# Folder containing the cleaned images.
CLEAN_IMAGE_FOLDER = Path("Data/processed/clean_images")

# Folder where the embeddings will be saved.
ARTIFACT_FOLDER = Path("artifacts")

# File containing all image embeddings.
EMBEDDINGS_FILE = ARTIFACT_FOLDER / "image_embeddings.npy"

# File containing image paths in matching order.
IMAGE_PATHS_FILE = ARTIFACT_FOLDER / "image_paths.npy"

# Number of images processed at one time.
BATCH_SIZE = 16


def get_device():
    """
    Select the best available processing device.
    """

    # Use an NVIDIA GPU if available.
    if torch.cuda.is_available():
        return "cuda"

    # Use the Apple GPU if available.
    if torch.backends.mps.is_available():
        return "mps"

    # Otherwise use the CPU.
    return "cpu"


def get_image_paths():
    """
    Return all cleaned image paths in sorted order.
    """

    # File extensions that are accepted as images.
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    # Find all image files in the cleaned folder.
    image_paths = [
        image_path
        for image_path in CLEAN_IMAGE_FOLDER.iterdir()
        if image_path.is_file()
        and image_path.suffix.lower() in image_extensions
    ]

    # Sort paths so embedding order is consistent.
    image_paths.sort()

    # Stop if no images are found.
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {CLEAN_IMAGE_FOLDER}"
        )

    # Return the sorted image paths.
    return image_paths


def create_batch_embeddings(
    batch_paths,
    model,
    processor,
    device
):
    """
    Create one CLIP embedding for every image in a batch.
    """

    # Store successfully opened images.
    images = []

    # Store paths that match the successfully opened images.
    valid_paths = []

    # Open each image in the current batch.
    for image_path in batch_paths:
        try:
            # Open the image and convert it to RGB.
            image = Image.open(image_path).convert("RGB")

            # Add the image to the current batch.
            images.append(image)

            # Store the matching path.
            valid_paths.append(image_path)

        except Exception as error:
            # Report and skip an image that cannot be opened.
            print(f"Skipping {image_path}: {error}")

    # Return empty results if no images opened successfully.
    if not images:
        return None, []

    # Convert the images into CLIP input tensors.
    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    # Move the input tensors to the selected device.
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Disable gradients because the model is not being trained.
    with torch.no_grad():

        # Generate the image features.
        output = model.get_image_features(
            **inputs,
            return_dict=False
        )

    # Handle tuple-style output from some Transformers versions.
    if isinstance(output, tuple):
        embeddings = output[0]
    else:
        embeddings = output

    # If multiple images are processed, embeddings should be:
    # (batch_size, 768).
    # If the model returns visual tokens, average those tokens.
    if embeddings.ndim == 3:
        embeddings = embeddings.mean(dim=1)

    # Normalise every embedding for cosine similarity.
    embeddings = embeddings / embeddings.norm(
        dim=-1,
        keepdim=True
    )

    # Convert the batch embeddings to NumPy arrays.
    embeddings = embeddings.cpu().numpy()

    # Return embeddings and matching image paths.
    return embeddings, valid_paths


def create_catalogue_embeddings():
    """
    Process all cleaned images and save their embeddings.
    """

    # Select the best available device.
    device = get_device()

    # Display the selected device.
    print(f"Using device: {device}")

    # Load the CLIP processor.
    print("Loading CLIP processor...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # Load the CLIP model.
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME)

    # Move the model to the selected device.
    model.to(device)

    # Put the model into evaluation mode.
    model.eval()

    # Get all cleaned image paths.
    image_paths = get_image_paths()

    # Create the artifacts folder.
    ARTIFACT_FOLDER.mkdir(parents=True, exist_ok=True)

    # Store all generated embedding batches.
    all_embeddings = []

    # Store paths matching the embeddings.
    all_valid_paths = []

    # Process the images in batches.
    for start_index in tqdm(
        range(0, len(image_paths), BATCH_SIZE),
        desc="Creating catalogue embeddings"
    ):

        # Select the current batch.
        batch_paths = image_paths[
            start_index:start_index + BATCH_SIZE
        ]

        # Create embeddings for this batch.
        batch_embeddings, valid_paths = create_batch_embeddings(
            batch_paths=batch_paths,
            model=model,
            processor=processor,
            device=device
        )

        # Skip a batch if none of its images were valid.
        if batch_embeddings is None:
            continue

        # Store this batch's embeddings.
        all_embeddings.append(batch_embeddings)

        # Store this batch's matching paths.
        all_valid_paths.extend(valid_paths)

    # Combine all batches into one array.
    embeddings = np.vstack(all_embeddings)

    # Save all embeddings to disk.
    np.save(EMBEDDINGS_FILE, embeddings)

    # Save matching image paths to disk.
    np.save(
        IMAGE_PATHS_FILE,
        np.array([str(path) for path in all_valid_paths])
    )

    # Display the final result.
    print("\n--- Catalogue Embedding Summary ---")
    print(f"Images processed: {len(all_valid_paths)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embeddings saved to: {EMBEDDINGS_FILE}")
    print(f"Image paths saved to: {IMAGE_PATHS_FILE}")


# Run the catalogue process when this file is executed directly.
if __name__ == "__main__":
    create_catalogue_embeddings()