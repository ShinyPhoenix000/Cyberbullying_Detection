# Cyber Bullying Detection

## Overview
This project analyzes social media comments from CSV files to detect possible cyberbullying. It uses text preprocessing and sentiment analysis to classify comments as negative, neutral, or positive, and flags cases with a high proportion of negative comments.

## Main Workflow
1. **Data Input**: Loads comment data from CSV files (e.g., `twitter_data.csv`, `twitter100_data.csv`, `ytcomments100list.csv`).
2. **Preprocessing**: Cleans comments (lowercasing, removing punctuation & stopwords) via NLTK in `text_preprocessor.py`.
3. **Detection**: Uses TextBlob for sentiment polarity → classifies as negative, neutral, positive → compares proportion of negative comments to a threshold (15% or 40%).
4. **Outputs**: Prints summary stats, logs to `log.csv` and `alert_log.csv`, and saves processed data in `processed_comments.csv`.

## Features
- **Alert Logging**: Logs alerts to `alert_log.csv` and general logs to `log.csv`.
- **Visualization**: Uses Matplotlib & Seaborn (`textwm.py`, `unlabelledmtb.py`).
- **Email Notifications**: Sends alerts when cyberbullying is detected (`textwm.py`).
- **Accuracy Calculation**: Compares predicted sentiment to true labels (`mtb.py`).

## Suggested Improvements
- Secure email credentials.
- Configurable thresholds (via config file or CLI).
- Better modularization (functions/classes).
- Add unit tests.
- Support more platforms/data formats.
- Upgrade to ML-based models.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download NLTK resources (done automatically, but you can run manually):
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```
3. Place your CSV data file (e.g., `twitter_data.csv`) in the project folder.
4. Run the main script:
   ```bash
   python mtb.py
   # or
   python textwm.py
   ```
5. Check the console output for summary and alerts.
6. View logs in `log.csv` and `alert_log.csv`.
7. If using `textwm.py`, check your email for notifications and view the sentiment plot.

## Advanced/Unique Aspects
- Uses rule-based (TextBlob) and threshold-based detection.
- Automated alerting via email.
- Visualization for quick insight.
- Modular preprocessing for text cleaning.


Feel free to contribute or suggest improvements!

## License

This project is licensed under the [MIT License](./LICENSE).<br>
Copyright © 2025 Kona Shiny Phoenix.
