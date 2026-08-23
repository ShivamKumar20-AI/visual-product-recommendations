# Import Path to work with image file paths.
from pathlib import Path

# Import the catalogue-loading and recommendation functions.
from recommender import (
    load_catalogue_data,
    find_similar_images
)

# Import the CLIP embedding function and model settings.
from feature_extraction import (
    get_device,
    create_one_embedding,
    MODEL_NAME
)

# Import PyTorch to load the CLIP model.
import torch

# Import the CLIP model and processor.
from transformers import CLIPModel, CLIPProcessor


# Use one cleaned image as the test query image.
TEST_IMAGE = Path(
    "Data/processed/clean_images/image_000001.jpg"
)


def test_recommendations():
    """
    Generate five recommendations for one test image.
    """

    # Select CPU, CUDA, or Apple MPS.
    device = get_device()

    # Load the CLIP processor.
    print("Loading CLIP processor...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # Load the pre-trained CLIP model.
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(MODEL_NAME)

    # Move the model to the selected device.
    model.to(device)

    # Put the model into evaluation mode.
    model.eval()

    # Create an embedding for the test image.
    query_embedding = create_one_embedding(
        image_path=TEST_IMAGE,
        model=model,
        processor=processor,
        device=device
    )

    # Load the saved catalogue embeddings and image paths.
    catalogue_paths, catalogue_embeddings = load_catalogue_data()

    # Find the five most similar images, excluding the query image itself.
    recommendations = find_similar_images(
        query_embedding=query_embedding,
        catalogue_embeddings=catalogue_embeddings,
        catalogue_paths=catalogue_paths,
        number_of_recommendations=5,
        query_path=TEST_IMAGE,
    )

    # Display the results.
    print("\n--- Recommendation Test ---")
    print(f"Query image: {TEST_IMAGE}")

    for position, recommendation in enumerate(
        recommendations,
        start=1
    ):
        print(
            f"{position}. "
            f"{recommendation['image_path']} | "
            f"Similarity: "
            f"{recommendation['similarity_score']:.4f}"
        )


# Run the test when this file is executed directly.
if __name__ == "__main__":
    test_recommendations()