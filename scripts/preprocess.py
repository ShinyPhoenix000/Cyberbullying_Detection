import re
import spacy
import emoji
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already done
nltk.download('stopwords')

# Load Spacy English model
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """
    Clean and preprocess text for ML models.
    
    Steps:
    1. Lowercase
    2. Remove emojis
    3. Remove mentions (@user)
    4. Remove hashtags (#topic)
    5. Remove URLs
    6. Remove punctuation
    7. Lemmatize and remove stopwords
    """
    # Lowercase
    text = text.lower()

    # Remove emojis
    text = emoji.replace_emoji(text, replace='')

    # Remove mentions, hashtags, URLs
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'http\S+', '', text)

    # Remove punctuation (basic version, works with standard re)
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize and lemmatize with spaCy
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_space]

    return ' '.join(tokens)