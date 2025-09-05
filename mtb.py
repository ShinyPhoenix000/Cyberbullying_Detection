nltk.download('punkt_tab')
import pandas as pd
from text_preprocessor import preprocess_text
from textblob import TextBlob
from datetime import datetime
import csv
import nltk
from sklearn.metrics import accuracy_score
import argparse
import os


nltk.download('punkt_tab')

# Function to log alert messages to a separate CSV file
def log_alert_message(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Check if the alert CSV file exists; if not, create it and write headers
    try:
        with open('alert_log.csv', 'x', newline='') as file:  # 'x' mode creates the file if it doesn't exist
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'message'])  # Write headers
    except FileExistsError:
        pass  # If the file exists, no need to write headers again

    with open('alert_log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, message])

# Function to log messages to a general CSV file (existing function)
def log_message_csv(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Check if the CSV file exists; if not, create it and write headers
    try:
        with open('log.csv', 'x', newline='') as file:  # 'x' mode creates the file if it doesn't exist
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'message'])  # Write headers
    except FileExistsError:
        pass  # If the file exists, no need to write headers again

    with open('log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, message])


def main():
    parser = argparse.ArgumentParser(description="Cyber Bullying Detection")
    parser.add_argument('--input', type=str, default="data/twitter_data.csv", help="Path to input CSV file")
    parser.add_argument('--output', type=str, default="results/", help="Path to results directory")
    parser.add_argument('--config', type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Load the dataset
    df = pd.read_csv(args.input)

    # Preprocess the comments
    df['cleaned_comment'] = df['comment'].apply(preprocess_text)

    # Check for NaN values in 'label' and drop rows with NaN values
    df = df.dropna(subset=['label'])

    # Function to calculate polarity of the comment
    def get_polarity(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity

    # Apply polarity calculation
    df['polarity'] = df['cleaned_comment'].apply(get_polarity)

    # Function to classify polarity into sentiment categories
    def classify_sentiment(polarity):
        if polarity < 0:
            return -1  # Negative sentiment
        elif polarity == 0:
            return 0  # Neutral sentiment
        else:
            return 1  # Positive sentiment

    # Get predicted sentiments based on polarity
    df['predicted_sentiment'] = df['polarity'].apply(classify_sentiment)

    # Compare predicted sentiments with the true labels (from the 'label' column)
    true_labels = df['label'].values
    predicted_labels = df['predicted_sentiment'].values

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Print out the accuracy result
    print(f"TextBlob Accuracy: {accuracy:.2f}")

    # Count negative and neutral comments
    negative_count = df['predicted_sentiment'].apply(lambda x: x == -1).sum()
    neutral_count = df['predicted_sentiment'].apply(lambda x: x == 0).sum()
    positive_count = df['predicted_sentiment'].apply(lambda x: x == 1).sum()

    # Print results
    print(f"\nTotal Comments: {len(df)}")
    print(f"Negative Comments: {negative_count}")
    print(f"Neutral Comments: {neutral_count}")
    print(f"Positive Comments: {positive_count}")

    # Calculate threshold (e.g., 15% of total comments)
    threshold = 0.15 * len(df)
    flagged = negative_count > threshold

    # Print results to console
    print(f"Threshold: {threshold}\n")
    if flagged:
        print("Warning: This may be a case of cyberbullying!")
    else:
        print("No significant negative activity detected.")

    # Log the general message (both for logging and alerting)
    log_message_csv(f"Negative Comments: {negative_count} out of {len(df)}")

    # Log the alert message based on the flag
    if flagged:
        log_message_csv(f"Warning: Cyberbullying detected. {negative_count} negative comments out of {len(df)}.")
        log_alert_message(f"Warning: Cyberbullying detected. {negative_count} negative comments out of {len(df)}.")  # Log to separate file
    else:
        log_message_csv(f"No significant negative activity detected. {negative_count} negative comments out of {len(df)}.")
        log_alert_message(f"No significant negative activity detected. {negative_count} negative comments out of {len(df)}.")  # Log to separate file

    # Save results to CSV in output directory
    output_csv = os.path.join(args.output, 'processed_comments.csv')
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
