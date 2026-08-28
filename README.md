## Project structure

```text
visual-product-recommendations/
├── app/streamlit_app.py          # User interface
├── src/
│   ├── data_cleaning.py          # Removes invalid and duplicate images
│   ├── feature_extraction.py     # CLIP image preprocessing
│   ├── create_catalogue_embeddings.py
│   ├── recommender.py            # Cosine similarity search
│   └── evaluate.py               # Test results and visualisation
├── artifacts/
│   ├── image_embeddings.npy      # Saved CLIP image vectors
│   └── image_paths.npy           # Matching image locations
├── Data/processed/clean_images/  # Clean product catalogue
├── run_pipeline.py               # Runs preparation pipeline
├── requirements.txt
└── README.md
```

## Run locally

```bash
git clone https://github.com/ShivamKumar20-AI/visual-product-recommendations.git
cd visual-product-recommendations

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
```

Add the downloaded dataset to:

```text
raw/archive/images/
```

Build the clean catalogue and CLIP embeddings:

```bash
python run_pipeline.py
```

Launch the app:

```bash
streamlit run app/streamlit_app.py
```
