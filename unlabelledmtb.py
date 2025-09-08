import pandas as pd
from text_preprocessor import preprocess_text
from textblob import TextBlob
from datetime import datetime
import csv
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

nltk.download('punkt_tab')

def get_polarity(text):
    """
    Calculate polarity of a comment using TextBlob.
    Polarity ranges from -1 (negative) to 1 (positive).
    """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def is_negative(polarity):
    """
    Classify a comment as negative if polarity < 0.
    """
    return polarity < 0

def log_message_csv(message):
    """Log messages to CSV file with timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open('log.csv', 'x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'message'])
    except FileExistsError:
        pass

    with open('log.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, message])

def plot_sentiment_distribution(df):
    """Plot sentiment distribution using seaborn."""
    sentiment_counts = df['is_negative'].value_counts()
    sentiment_counts = sentiment_counts.sort_index()
    plt.figure(figsize=(6, 6))
    sns.barplot(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
    )

def main():
    df = pd.read_csv('ytcomments100list.csv')
    df['cleaned_comment'] = df['comment'].apply(preprocess_text)
    df['polarity'] = df['cleaned_comment'].apply(get_polarity)
    df['is_negative'] = df['polarity'].apply(is_negative)

    negative_count = df['is_negative'].sum()
    threshold = 0.4 * len(df)
    flagged = negative_count > threshold

    print(f"\nTotal Comments: {len(df)}")
    print(f"Negative Comments: {negative_count}")
    print(f"Threshold: {threshold}\n")
    
    if flagged:
        print("Warning: This may be a case of cyberbullying!")
        log_message_csv(f"Warning: Cyberbullying detected. {negative_count} negative comments out of {len(df)}.")
    else:
        print("No significant negative activity detected.")
        log_message_csv(f"No significant negative activity detected. {negative_count} negative comments out of {len(df)}.")

    df.to_csv('processed_comments.csv', index=False)

if __name__ == "__main__":
    main()
        hue=sentiment_counts.index,  # Assign x to hue
        palette='Blues', 
        dodge=False  # Optional, avoids separate bars for hue
    )
    plt.title('Sentiment Distribution of Comments')
    plt.xlabel('Sentiment (Negative/Positive)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Positive', 'Negative'], rotation=45)
    plt.legend([], [], frameon=False)  # Disable legend
    plt.show()


# Call the functions to display the summary and plot the distribution
# display_summary(df)
plot_sentiment_distribution(df)