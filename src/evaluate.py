# Import Path so we can work with image file paths.
from pathlib import Path

# Import Matplotlib to create and save the recommendation figure.
import matplotlib.pyplot as plt

# Import Pillow to open images for display.
from PIL import Image

# Import PyTorch to load the CLIP model.
import torch

# Import the CLIP model and processor.
from transformers import CLIPModel, CLIPProcessor

# Import functions from our recommendation file.
from recommender import (
    load_catalogue_data,
    find_similar_images
)

# Import functions from our feature extraction file.
from feature_extraction import (
    get_device,
    create_one_embedding,
    MODEL_NAME
)


# Image used as the query for this evaluation.
TEST_IMAGE = Path(
    "Data/processed/clean_images/image_000001.jpg"
)

# Location where the final figure will be saved.
OUTPUT_FIGURE = Path(
    "artifacts/recommendation_result.png"
)


def create_recommendation_figure():
    """
    Create a figure showing the query image and five recommendations.
    """

    # Select CPU, CUDA, or Apple MPS.
    device = get_device()

    # Display the selected device.
    print(f"Using device: {device}")

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

    # Create an embedding for the query image.
    query_embedding = create_one_embedding(
        image_path=TEST_IMAGE,
        model=model,
        processor=processor,
        device=device
    )

    # Load the saved catalogue embeddings and matching paths.
    catalogue_paths, catalogue_embeddings = load_catalogue_data()

    # Find five recommendations, excluding the query image.
    recommendations = find_similar_images(
        query_embedding=query_embedding,
        catalogue_embeddings=catalogue_embeddings,
        catalogue_paths=catalogue_paths,
        number_of_recommendations=5,
        query_path=TEST_IMAGE
    )

    # Create the output folder if it does not exist.
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)

    # Create a 2-by-3 grid of image panels.
    figure, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(15, 9)
    )

    # Flatten the axes so they are easier to access.
    axes = axes.flatten()

    # Display the query image in the first panel.
    query_image = Image.open(TEST_IMAGE).convert("RGB")

    axes[0].imshow(query_image)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    # Display each recommendation in the remaining panels.
    for position, recommendation in enumerate(
        recommendations,
        start=1
    ):
        # Open the recommended image.
        recommended_image = Image.open(
            recommendation["image_path"]
        ).convert("RGB")

        # Display the recommended image.
        axes[position].imshow(recommended_image)

        # Read the similarity score.
        score = recommendation["similarity_score"]

        # Display the rank and score.
        axes[position].set_title(
            f"Recommendation {position}\n"
            f"Cosine similarity: {score:.4f}"
        )

        # Hide the axes around the image.
        axes[position].axis("off")

    # Add an overall title to the figure.
    figure.suptitle(
        "Visual Clothing Recommendations Using CLIP",
        fontsize=16
    )

    # Adjust spacing so titles do not overlap.
    figure.tight_layout()

    # Save the figure as a PNG file.
    figure.savefig(
        OUTPUT_FIGURE,
        dpi=200,
        bbox_inches="tight"
    )

    # Close the figure to release memory.
    plt.close(figure)

    # Display the output location and recommendation scores.
    print("\n--- Evaluation Complete ---")
    print(f"Figure saved to: {OUTPUT_FIGURE}")

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


# Run the evaluation when this file is executed directly.
if __name__ == "__main__":
    create_recommendation_figure()