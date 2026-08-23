# Import PyTorch to run the CLIP model.
import torch

# Import the CLIP model and image processor.
from transformers import CLIPModel, CLIPProcessor

# Import Image from Pillow to open uploaded images.
from PIL import Image

# Import sys so we can add the project root to Python's import path.
import sys

# Import Path to create reliable folder paths.
from pathlib import Path

# Find the main project folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add the project folder so Python can find the src package.
sys.path.insert(0, str(PROJECT_ROOT))

# Import Streamlit to create the web application.
import streamlit as st

from src.recommender import (
    load_catalogue_data,
    find_similar_images
)

from src.feature_extraction import (
    get_device,
    create_one_embedding,
    MODEL_NAME
)


# Set the title shown in the browser.
st.set_page_config(
    page_title="Visual Clothing Recommendations",
    page_icon="👕",
    layout="wide"
)


@st.cache_resource
def load_model():
    """
    Load CLIP once and reuse it between uploads.
    """

    # Select the best available processing device.
    device = get_device()

    # Load the CLIP processor.
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    # Load the pre-trained CLIP model.
    model = CLIPModel.from_pretrained(MODEL_NAME)

    # Move the model to the selected device.
    model.to(device)

    # Put the model into evaluation mode.
    model.eval()

    # Return the model components.
    return model, processor, device


@st.cache_data
def load_saved_catalogue():
    """
    Load the saved image embeddings and paths once.
    """

    # Load the saved catalogue data.
    catalogue_paths, catalogue_embeddings = load_catalogue_data()

    # Return the saved data.
    return catalogue_paths, catalogue_embeddings


def main():
    """
    Run the Streamlit application.
    """

    # Display the application heading.
    st.title("Visual Clothing Recommendations")

    # Explain how the app works.
    st.write(
        "Upload an image of clothing to find five visually similar "
        "items from the retailer catalogue."
    )

    # Create the upload control.
    uploaded_file = st.file_uploader(
        "Upload a clothing image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    # Stop until the user uploads an image.
    if uploaded_file is None:
        st.info("Please upload a clothing image to begin.")
        return

    # Open the uploaded image.
    uploaded_image = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded image.
    st.subheader("Your uploaded image")
    st.image(
        uploaded_image,
        width=300
    )

    # Load the CLIP model and processor.
    with st.spinner("Loading the image model..."):
        model, processor, device = load_model()

    # Create a temporary path-like value for the uploaded image.
    # The uploaded image itself is passed directly to CLIP below.
    inputs = processor(
        images=uploaded_image,
        return_tensors="pt"
    )

    # Move the processor output to the selected device.
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # Create the uploaded image embedding.
    with torch.no_grad():
        image_output = model.get_image_features(
            **inputs,
            return_dict=False
        )

    # Handle tuple-style model output.
    if isinstance(image_output, tuple):
        embedding_tensor = image_output[0]
    else:
        embedding_tensor = image_output

    # If visual tokens are returned, average them into one image vector.
    if embedding_tensor.ndim == 3:
        embedding_tensor = embedding_tensor.mean(dim=1)

    # Remove the batch dimension.
    query_embedding = embedding_tensor[0]

    # Move the embedding to the CPU and convert it to NumPy.
    query_embedding = query_embedding.cpu().numpy()

    # Normalise the uploaded image embedding.
    embedding_length = (query_embedding ** 2).sum() ** 0.5
    query_embedding = query_embedding / embedding_length

    # Load the saved catalogue embeddings and paths.
    catalogue_paths, catalogue_embeddings = load_saved_catalogue()

    # Find the five closest catalogue images.
    recommendations = find_similar_images(
        query_embedding=query_embedding,
        catalogue_embeddings=catalogue_embeddings,
        catalogue_paths=catalogue_paths,
        number_of_recommendations=5
    )

    # Display the recommendation heading.
    st.subheader("Recommended clothing")

    # Create five columns for the recommendations.
    columns = st.columns(5)

    # Display each recommended image in its own column.
    for column, recommendation in zip(
        columns,
        recommendations
    ):
        # Get the recommended image path.
        image_path = Path(recommendation["image_path"])

        # Open the recommended image.
        recommended_image = Image.open(image_path).convert("RGB")

        # Read the similarity score.
        similarity_score = recommendation["similarity_score"]

        # Display the image and score.
        with column:
            st.image(
                recommended_image,
                use_container_width=True
            )

            st.write(
                f"Cosine similarity: {similarity_score:.4f}"
            )


# Start the Streamlit app.
if __name__ == "__main__":
    main()