# Import Path so we can work with folders and file paths.
from pathlib import Path

# Import NumPy for loading embeddings and calculating similarity.
import numpy as np


# Folder containing the cleaned clothing images.
CLEAN_IMAGE_FOLDER = Path("Data/processed/clean_images")

# File containing the saved catalogue embeddings.
EMBEDDINGS_FILE = Path("artifacts/image_embeddings.npy")

# File containing image paths in matching embedding order.
IMAGE_PATHS_FILE = Path("artifacts/image_paths.npy")


def load_catalogue_data():
    """
    Load and validate the saved catalogue embeddings and image paths.
    """

    # Check that the embeddings file exists.
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDINGS_FILE}"
        )

    # Check that the image paths file exists.
    if not IMAGE_PATHS_FILE.exists():
        raise FileNotFoundError(
            f"Image paths file not found: {IMAGE_PATHS_FILE}"
        )

    # Load the complete embedding matrix.
    catalogue_embeddings = np.load(EMBEDDINGS_FILE)

    # Load image paths in the same order as the embeddings.
    catalogue_paths = np.load(IMAGE_PATHS_FILE)

    # The embeddings must be a two-dimensional matrix.
    # Expected shape: (31378, 768).
    if catalogue_embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must have two dimensions: "
            "(number_of_images, embedding_size). "
            f"Received shape: {catalogue_embeddings.shape}"
        )

    # Confirm that each embedding has 768 values.
    if catalogue_embeddings.shape[1] != 768:
        raise ValueError(
            "Each embedding should contain 768 values. "
            f"Received shape: {catalogue_embeddings.shape}"
        )

    # Confirm that the number of paths matches the number of embeddings.
    if len(catalogue_embeddings) != len(catalogue_paths):
        raise ValueError(
            "The number of embeddings does not match "
            "the number of image paths."
        )

    # Return paths first and embeddings second.
    return catalogue_paths, catalogue_embeddings


def calculate_cosine_similarity(embedding_a, embedding_b):
    """
    Calculate the cosine similarity between two embeddings.
    """

    # Convert both inputs into NumPy arrays.
    embedding_a = np.asarray(embedding_a)
    embedding_b = np.asarray(embedding_b)

    # Calculate the length of each embedding.
    length_a = np.linalg.norm(embedding_a)
    length_b = np.linalg.norm(embedding_b)

    # Prevent division by zero.
    if length_a == 0 or length_b == 0:
        raise ValueError(
            "Cosine similarity cannot be calculated for a zero vector."
        )

    # Calculate the dot product divided by both vector lengths.
    similarity = np.dot(embedding_a, embedding_b) / (
        length_a * length_b
    )

    # Return a normal Python float.
    return float(similarity)


def calculate_all_similarities(
    query_embedding,
    catalogue_embeddings
):
    """
    Calculate similarity between one query embedding
    and every catalogue embedding.
    """

    # Convert the query embedding to a NumPy array.
    query_embedding = np.asarray(query_embedding)

    # Confirm that the query is a one-dimensional vector.
    if query_embedding.ndim != 1:
        raise ValueError(
            "The query embedding must have shape (768,). "
            f"Received shape: {query_embedding.shape}"
        )

    # Calculate the length of the query embedding.
    query_length = np.linalg.norm(query_embedding)

    # Prevent division by zero.
    if query_length == 0:
        raise ValueError(
            "The query embedding cannot be a zero vector."
        )

    # Normalise the query embedding.
    normalised_query = query_embedding / query_length

    # Calculate the length of every catalogue embedding.
    catalogue_lengths = np.linalg.norm(
        catalogue_embeddings,
        axis=1,
        keepdims=True
    )

    # Prevent division by zero in the catalogue.
    if np.any(catalogue_lengths == 0):
        raise ValueError(
            "At least one catalogue embedding is a zero vector."
        )

    # Normalise every catalogue embedding.
    normalised_catalogue = (
        catalogue_embeddings / catalogue_lengths
    )

    # Compare the query against every catalogue image.
    # The result contains one score per catalogue image.
    similarities = normalised_catalogue @ normalised_query

    # Return the similarity scores.
    return similarities


def find_similar_images( # Return the top N similar catalogue images.
    query_embedding,
    catalogue_embeddings,
    catalogue_paths,
    number_of_recommendations=5,
    query_path=None
):

    # Calculate similarity against every catalogue image.
    similarities = calculate_all_similarities(
        query_embedding=query_embedding,
        catalogue_embeddings=catalogue_embeddings
    )

    # If the query image is already in the catalogue, exclude it.
    if query_path is not None:

        # Convert the query path to text for a reliable comparison.
        query_path_text = str(query_path)

        # Set its score to negative infinity so it cannot be selected.
        for index, catalogue_path in enumerate(catalogue_paths):

            # Compare the saved catalogue path with the query path.
            if str(catalogue_path) == query_path_text:
                similarities[index] = -np.inf

    # Sort from highest similarity to lowest similarity.
    ranked_indexes = np.argsort(similarities)[::-1]

    # Select the requested number of recommendations.
    top_indexes = ranked_indexes[:number_of_recommendations]

    # Store the final results.
    recommendations = []

    # Build one result for each selected image.
    for index in top_indexes:
        recommendations.append({
            "image_path": Path(catalogue_paths[index]),
            "similarity_score": float(similarities[index])
        })

    # Return the complete recommendation list.
    return recommendations

def test_catalogue_loading():
    """
    Test that the saved embeddings and paths load correctly.
    """

    # Load the catalogue data.
    catalogue_paths, catalogue_embeddings = load_catalogue_data()

    # Print the shapes and counts.
    print("\n--- Catalogue Loading Test ---")
    print(f"Embedding shape: {catalogue_embeddings.shape}")
    print(f"Number of image paths: {len(catalogue_paths)}")

    # Confirm that the two files match.
    if len(catalogue_embeddings) != len(catalogue_paths):
        print("Test failed: embeddings and paths do not match.")
    else:
        print("Test passed: embeddings and paths match.")


# Run the catalogue loading test when this file is executed directly.
if __name__ == "__main__":
    test_catalogue_loading()