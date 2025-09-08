import nltk
from nltk.corpus import stopwords
import string

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess a given text by:
    - Converting to lowercase
    - Removing punctuation
    - Removing stop words
    """
    if not isinstance(text, str):
        return ''
        
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)