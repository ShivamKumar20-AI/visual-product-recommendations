README.md content:

# Visual Product Recommendations

A simple AI app that recommends clothing items that look similar to an uploaded clothing image.

## Live app

Use the app here:
https://visual-appuct-recommendations.streamlit.app/

## What the app does

1. Upload an image of a clothing item.
2. The app turns the image into a list of numbers called an image embedding.
3. It compares the uploaded image with the catalogue images.
4. It returns the five most visually similar products.
5. It shows a cosine similarity score for every recommendation.

A higher score means the images are more similar. A score closer to 1 means stronger similarity.

## Model

The project uses the pretrained CLIP image model, `openai/clip-vit-base-patch32`, through Hugging Face Transformers. CLIP converts both the uploaded image and catalogue images into comparable image embeddings. The app then uses cosine similarity to rank the catalogue images.

This is an unsupervised image-recommendation approach because the system does not need product labels or a target variable to make recommendations.

## Data preparation

The image catalogue was cleaned before modelling. Invalid images and exact duplicate images were removed. The final catalogue used by the model contains 5,017 images.

The saved catalogue files are:

- `artifacts/image_embeddings.npy`: saved image embeddings.
- `artifacts/image_paths.npy`: paths to the catalogue images.

## Scores and results

The model was tested using the recommendation function and returned five recommendations for a test image. The cosine similarity scores were:

| Recommendation | Cosine similarity |
|---|---:|
| 1 | 0.7791 |
| 2 | 0.7746 |
| 3 | 0.7722 |
| 4 | 0.7698 |
| 5 | 0.7679 |
|
| Average of top five | 0.7727 |

All five scores were above 0.7. This indicates that the recommendations were visually similar to the test image.

The score range was 0.7679 to 0.7791, giving a spread of 0.0112 between the lowest and highest recommendation.

The result plot is saved as `artifacts/recommendation_result.png`.

## Project structure

```text
visual-product-recommendations/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── data_cleaning.py
│   ├── feature_extraction.py
│   ├── create_catalogue_embeddings.py
│   ├── recommender.py
│   ├── evaluate.py
│   └── test_recommender.py
├── artifacts/
│   ├── image_embeddings.npy
│   └── image_paths.npy
├── Data/processed/clean_images/
├── requirements.txt
└── README.md
```

## Run the project locally

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows, activate it with:

```bash
.venv\\Scripts\\activate
```

Install the packages:

```bash
pip install -r requirements.txt
```

Start the app:

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser.

## Limitations

- The model recommends items based on visual appearance only.
- It does not understand price, size, colour preference, stock availability, or brand preference.
- Cosine similarity measures visual closeness; it is not a human rating of recommendation quality.
- The model uses a pretrained CLIP model and was not fine-tuned on this specific clothing catalogue.
- The catalogue is limited to the available cleaned images.

## Possible improvements

- Fine-tune the model using clothing-specific data.
- Add filters for category, colour, size, price, and availability.
- Test the model on more images and report average results across the test set.
- Add product names, descriptions, and links to the recommendations.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- CLIP
- NumPy
- Pillow
- Streamlit

## Author

Shivam Kumar
