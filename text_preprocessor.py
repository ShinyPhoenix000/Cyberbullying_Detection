import nltk
from nltk.corpus import stopwords
import string

# Download necessary NLTK resources if they aren't already downloaded
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Define stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess a given text by:
    - Converting to lowercase
    - Removing punctuation
    - Removing stop words
    """
    # Ensure the input is a string
    if not isinstance(text, str):
        return ''

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation using str.translate()
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize the text (split into individual words)
    words = nltk.word_tokenize(text)

    # Remove stop words
    filtered_words = [word for word in words if word not in stop_words]

    # Join the words back into a string and return it
    return ' '.join(filtered_words)