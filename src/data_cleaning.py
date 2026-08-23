# Import Path to work with folders and file paths.
from pathlib import Path

# Import shutil to copy image files.
import shutil

# Import hashlib to detect exact duplicate file contents.
import hashlib

# Import Image from Pillow to test whether images are valid.
from PIL import Image


# Original dataset image folder.
RAW_IMAGE_FOLDER = Path("raw/archive/images")

# Folder where valid, unique images will be copied.
CLEAN_IMAGE_FOLDER = Path("Data/processed/clean_images")

# File extensions accepted as images.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_file_hash(file_path): # Return a fingerprint for a file to identify exact duplicates.
    

    # Create a fingerprint object.
    file_hash = hashlib.md5()

    # Read the image file as bytes.
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            file_hash.update(chunk)

    # Return the completed fingerprint.
    return file_hash.hexdigest()


def is_valid_image(image_path): # Return True if Pillow can open and verify the image.
    

    try:
        # Check that the image file is readable and not corrupted.
        with Image.open(image_path) as image:
            image.verify()

        return True

    except Exception:
        return False


def clear_clean_folder(): # Delete previous cleaned output so the script starts fresh.
    

    # Delete the old cleaned folder if it exists.
    if CLEAN_IMAGE_FOLDER.exists():
        shutil.rmtree(CLEAN_IMAGE_FOLDER)

    # Create a new empty cleaned folder.
    CLEAN_IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)


def clean_images(): # Copy valid, unique images to the clean folder with unique filenames.
    

    # Check that the input folder exists before processing.
    if not RAW_IMAGE_FOLDER.exists():
        print(f"Error: Raw folder not found: {RAW_IMAGE_FOLDER}")
        return

    # Start with an empty output folder.
    clear_clean_folder()

    # Find all image files, including any within subfolders.
    image_files = [
        path
        for path in RAW_IMAGE_FOLDER.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Store file fingerprints that have already been processed.
    seen_hashes = set()

    # Set counters for the cleaning report.
    valid_unique_count = 0
    invalid_count = 0
    duplicate_count = 0

    # Process images one by one.
    for image_path in image_files:

        # Skip unreadable/corrupted images.
        if not is_valid_image(image_path):
            invalid_count += 1
            continue

        # Calculate an exact-content fingerprint for the image.
        image_hash = get_file_hash(image_path)

        # Skip the image if the exact same content was found before.
        if image_hash in seen_hashes:
            duplicate_count += 1
            continue

        # Remember this image fingerprint as unique.
        seen_hashes.add(image_hash)

        # Build a unique output filename.
        # Example: image_000001.jpg, image_000002.png
        output_name = f"image_{valid_unique_count + 1:06d}{image_path.suffix.lower()}"
        output_path = CLEAN_IMAGE_FOLDER / output_name

        # Copy the valid, unique image to the cleaned folder.
        shutil.copy2(image_path, output_path)

        # Increase the number of copied images.
        valid_unique_count += 1

    # Print the final report.
    print("\n--- Data Cleaning Summary ---")
    print(f"Images found: {len(image_files)}")
    print(f"Valid and unique images copied: {valid_unique_count}")
    print(f"Invalid images skipped: {invalid_count}")
    print(f"Duplicate images skipped: {duplicate_count}")
    print(f"Clean image folder: {CLEAN_IMAGE_FOLDER}")


# Run cleaning only when this Python file is executed directly.
if __name__ == "__main__":
    clean_images()