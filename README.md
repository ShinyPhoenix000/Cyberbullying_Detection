# Cyberbullying Detection System

A production-ready system for detecting and monitoring cyberbullying in text data using DistilBERT and Streamlit.

## Features

- Real-time cyberbullying detection using fine-tuned DistilBERT
- Interactive Streamlit dashboard with visualizations
- Email alerts for severe content detection
- CSV data import and preprocessing
- Model performance metrics and analysis
- Configurable alert thresholds

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Model Training

1. Prepare your training data in CSV format with columns:
   - `comment_text`: The text content
   - `label`: 0 (neutral), 1 (mild), or 2 (severe)

2. Run the training script:
```bash
python scripts/train_test_example.py \
    --data_path data/training_data.csv \
    --model_dir models/cyberbullying_model \
    --epochs 3 \
    --batch_size 16
```

## Running the Dashboard

1. Start the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

2. Upload your CSV file with comments to analyze

3. Configure email alerts in the sidebar (optional)
4. Start the API:
   ```bash
   uvicorn app:app --reload
   ```
5. Launch the dashboard:
   ```bash
   streamlit run dashboard.py
   ```

## License
MIT License © 2025 Kona Shiny Phoenix

This project is licensed under the [MIT License](./LICENSE).<br>
Copyright © 2025 Kona Shiny Phoenix.
