import pandas as pd
from text_preprocessor import preprocess_text
from textblob import TextBlob
from datetime import datetime
import csv
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from email.mime.image import MIMEImage

nltk.download('punkt_tab')

# Load the dataset
df = pd.read_csv('twitter100_data.csv')  # Assuming your updated CSV has the correct column names

# Preprocess the comments
df['cleaned_comment'] = df['comment'].apply(preprocess_text)

# Function to calculate polarity of the comment
def get_polarity(text):
    """
    Calculate polarity of a comment using TextBlob.
    Polarity ranges from -1 (negative) to 1 (positive).
    """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

# Apply polarity calculation
df['polarity'] = df['cleaned_comment'].apply(get_polarity)

# Determine if a comment is negative
def is_negative(polarity):
    """
    Classify a comment as negative if polarity < 0pyt.
    """
    return polarity < 0

# Add a column indicating negativity
df['is_negative'] = df['polarity'].apply(is_negative)

# Count negative comments
negative_count = df['is_negative'].sum()

# Calculate threshold (e.g., 40% of total comments)
threshold = 0.15 * len(df)
flagged = negative_count > threshold

# Print results to console
print(f"\nTotal Comments: {len(df)}")
print(f"Negative Comments: {negative_count}")
print(f"Threshold: {threshold}\n\n")
if flagged:
    print("Warning: This may be a case of cyberbullying!")
else:
    print("No significant negative activity detected.")

# Function to log messages to a CSV file
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

# Log the message based on the flag
if flagged:
    log_message_csv(f"Warning: Cyberbullying detected. {negative_count} negative comments out of {len(df)}.")
else:
    log_message_csv(f"No significant negative activity detected. {negative_count} negative comments out of {len(df)}.")

# Email automation for flagged cases
def send_email(subject, body, recipient_email):
    """
    Send an email alert.
    """
    sender_email = "bmsitraj@gmail.com"  # Replace with your email
    sender_password = "pxhj cmfe nnkv bqjf"  # Replace with your App Password securely

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    # Attach the body to the email
    msg.attach(MIMEText(body, 'plain'))

    # Connect to the email server and send the email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, sender_password)
            server.send_message(msg)
            print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# Send an email if flagged
if flagged:
    subject = "Cyberbullying Alert"
    body = (f"Warning: we have detected a case of cyberbullying in recent social media activity\n for professional help, please contact 797531251 or visit amahahealth.com\n")
    recipient_email = "bmsitraj@gmail.com"  # Replace with the recipient's email ****************
    send_email(subject, body, recipient_email)

# Save the results to a new CSV file (optional, for further analysis)
df.to_csv('processed_comments.csv', index=False)

# Plotting the sentiment distribution (Negative vs Positive Comments)
def plot_sentiment_distribution(df):
    sentiment_counts = df['is_negative'].value_counts()
    sentiment_counts = sentiment_counts.sort_index()  # Ensure consistent ordering of positive and negative

    plt.figure(figsize=(6, 6))
    sns.barplot(
        x=sentiment_counts.index, 
        y=sentiment_counts.values, 
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

# Call the plotting function
plot_sentiment_distribution(df)
