# Standard library imports
import os
import json
import string
import logging
import smtplib
from pathlib import Path
from email.mime.text import MIMEText

# Third-party imports
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix, classification_report

# Deep learning imports
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Cyberbullying Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Custom styling
st.markdown("""
    <style>
    .block-container {padding: 2rem}
    .st-emotion-cache-16idsys {padding-top: 2rem}
    .stPlotlyChart {margin: 1rem 0}
    div[data-testid="stMetricValue"] {font-size: 24px}
    </style>
    """, unsafe_allow_html=True)

# Import preprocessing function
from scripts.preprocess import clean_text

def load_email_config():
    """Load email configuration from JSON file"""
    config_path = Path('config/email_config.json')
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error("Invalid email configuration file")
        return {}

def save_email_config(config):
    """Save email configuration to JSON file"""
    config_path = Path('config/email_config.json')
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f)

def send_alert_email(subject, body, config):
    """Send email alert using configured SMTP settings"""
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = config['sender_email']
        msg['To'] = config['recipient_email']

        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            if config.get('use_tls', True):
                server.starttls()
            server.login(config['smtp_username'], config['smtp_password'])
            server.send_message(msg)
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False

# Test Mode for Development
if st.sidebar.checkbox('Enable Test Mode', False):
    test_comments = [
        {'text': 'You are amazing!', 'label': 0},
        {'text': 'I disagree with your opinion.', 'label': 0},
        {'text': 'You are the worst, nobody likes you.', 'label': 2},
        {'text': 'This is unacceptable behavior.', 'label': 1},
        {'text': 'Everyone hates you, loser.', 'label': 2}
    ]
    test_df = pd.DataFrame([(d['text'], d['label']) for d in test_comments], 
                          columns=['comment_text', 'label'])
    test_df['cleaned_text'] = test_df['comment_text'].apply(clean_text)
    df = pd.concat([df, test_df], ignore_index=True)
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from pathlib import Path
import json
import smtplib
from email.mime.text import MIMEText
import logging
import string

# Configure logging and suppress warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        return set(stopwords.words('english'))
    except Exception as e:
        logger.warning(f"Failed to download NLTK stopwords: {e}")
        return set()

stop_words = download_nltk_data()

# Set page configuration
st.set_page_config(
    page_title="Cyberbullying Detection Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# Custom styling
st.markdown("""
    <style>
    .block-container {padding: 2rem}
    .st-emotion-cache-16idsys {padding-top: 2rem}
    .stPlotlyChart {margin: 1rem 0}
    div[data-testid="stMetricValue"] {font-size: 24px}
    </style>
    """, unsafe_allow_html=True)

# Helper: Clean text (lowercase, remove punctuation, stopwords)
def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens).strip()

# Data Loading and Preprocessing
@st.cache_data
def load_and_preprocess_data(file_path=None):
    """Load and preprocess data from CSV file"""
    try:
        if file_path is None:
            st.info("👆 Please upload a CSV file to begin analysis")
            return None
            
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Handle common column name variations
        if 'comment' in df.columns and 'comment_text' not in df.columns:
            df = df.rename(columns={'comment': 'comment_text'})
        if 'target' in df.columns and 'label' not in df.columns:
            df = df.rename(columns={'target': 'label'})
            
        # Validate required columns
        if 'comment_text' not in df.columns:
            st.error("❌ Required column 'comment_text' not found in the dataset")
            return None
        
        # Handle cleaned_text column
        if 'cleaned_text' not in df.columns:
            st.info("🔄 Processing text data...")
            try:
                df['cleaned_text'] = df['comment_text'].astype(str).apply(clean_text)
                st.success("✅ Text preprocessing complete")
            except Exception as e:
                st.error(f"❌ Error during text preprocessing: {str(e)}")
                logger.error(f"Preprocessing error: {e}")
                return None
        else:
            st.success("✅ Using existing cleaned text data")
            
        # Ensure all required columns exist
        required_cols = ['comment_text', 'cleaned_text']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"❌ Missing required columns: {', '.join(missing)}")
            return None
            
        return df
        
    except Exception as e:
        logger.error(f"Data loading error: {e}")
        st.error(f"❌ Error loading data: {str(e)}")
        return None

# File Upload
st.sidebar.header("📁 Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload a CSV file containing comments for analysis"
)

# Sample size control
sample_size = st.sidebar.slider(
    "Sample Size",
    min_value=100,
    max_value=10000,
    value=1000,
    step=100,
    help="Number of comments to analyze"
)

# Data view options
show_raw = st.sidebar.checkbox(
    "Show Raw Data",
    value=False,
    help="Display the raw data before preprocessing"
)

# Load and process data
df = load_and_preprocess_data(uploaded_file)

if df is not None:
    # Sample data if needed
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    
    # Display raw data if requested
    if show_raw:
        st.subheader("📋 Raw Data Sample")
        st.dataframe(
            df[['comment_text', 'label']].head(),
            use_container_width=True
        )
        
# ===============================
# Alert Detection System
# ===============================
def detect_cyberbullying_alerts(df, threshold=0.15):
    """
    Detect potentially harmful comments based on severity threshold.
    Returns (has_alerts, flagged_df)
    """
    if df is None or 'predicted_label' not in df.columns:
        return False, pd.DataFrame()
        
    try:
        # Consider both numeric and string labels
        severe_mask = (df['predicted_label'] == 2) | (df['predicted_label'] == 'severe')
        severe_count = severe_mask.sum()
        
        # Calculate alert status
        has_alerts = False
        if len(df) > 0:
            alert_ratio = severe_count / len(df)
            has_alerts = alert_ratio > threshold
            
        # Get flagged comments if alerts exist
        flagged_df = df[severe_mask] if has_alerts else pd.DataFrame()
        
        return has_alerts, flagged_df
        
    except Exception as e:
        logger.error(f"Alert detection error: {e}")
        return False, pd.DataFrame()

def display_alerts(df, flagged_df, email_config):
    """Display alert status and flagged comments"""
    if df is None:
        return
        
    try:
        if not flagged_df.empty:
            st.error('⚠️ Cyberbullying Alerts Detected!')
            st.write("### Flagged Comments")
            
            # Prepare display columns
            display_cols = ['comment_text']
            if 'label' in flagged_df.columns:
                display_cols.append('label')
            if 'predicted_label' in flagged_df.columns:
                display_cols.append('predicted_label')
                
            # Style the dataframe
            def highlight_severity(val):
                return 'background-color: #ffcccc' if val in [2, 'severe'] else ''
                
            styled_df = (
                flagged_df[display_cols]
                .head(10)
                .style.applymap(highlight_severity, subset=['label', 'predicted_label'])
            )
            
            st.dataframe(styled_df)
            
            # Process email alerts if configured
            if process_alerts(flagged_df, email_config):
                st.success("✉️ Alert email sent successfully")
        else:
            st.success('✅ No cyberbullying alerts detected')
            
    except Exception as e:
        logger.error(f"Error displaying alerts: {e}")
        st.error("❌ Error processing alerts")

# Test Mode for Development
if df is not None and st.sidebar.checkbox('🔧 Enable Test Mode', False):
    st.sidebar.info("Test mode: Adding sample comments...")
    test_comments = [
        {'text': 'Great job on this!', 'label': 0},
        {'text': 'I respectfully disagree.', 'label': 0},
        {'text': 'This needs improvement.', 'label': 1},
        {'text': 'You are completely wrong!', 'label': 1},
        {'text': 'This is terrible, you should quit.', 'label': 2}
    ]
    
    # Create test dataframe with proper columns
    test_df = pd.DataFrame([
        (d['text'], d['label']) for d in test_comments
    ], columns=['comment_text', 'label'])
    
    # Apply same preprocessing to test data
    test_df['cleaned_text'] = test_df['comment_text'].astype(str).apply(clean_text)
    
    # Combine with main dataframe
    df = pd.concat([df, test_df], ignore_index=True)
    st.sidebar.success("✅ Added test data")# ===============================
# Test Mode: Add hardcoded negative comments for pipeline verification
test_mode = st.sidebar.checkbox('Test Mode: Add negative comments')
if test_mode:
    test_comments = [
        'You are the worst, nobody likes you.',
        'Go kill yourself.',
        'Everyone hates you, loser.',
        'You are so stupid and ugly.',
        'Why don’t you just disappear?'
    ]
    test_df = pd.DataFrame({'comment_text': test_comments, 'label': [2]*len(test_comments)})
    test_df['cleaned_text'] = test_df['comment_text'].apply(clean_text)
    df = pd.concat([df, test_df], ignore_index=True)


# Model Loading and Prediction
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def validate_model_files(model_path):
    """Check if all required model files exist"""
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json', 'special_tokens_map.json']
    model_path = Path(model_path)
    return all((model_path / f).exists() for f in required_files)

@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer with fallback options"""
    model_path = "models/distilbert_cyberbullying"
    fallback_model = "distilbert-base-uncased"
    
    try:
        if Path(model_path).exists() and validate_model_files(model_path):
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.eval()
            st.success("✅ Loaded fine-tuned cyberbullying detection model")
            return model, tokenizer
            
        st.warning("⚠️ Using fallback model - predictions may be less accurate")
        model = AutoModelForSequenceClassification.from_pretrained(
            fallback_model, 
            num_labels=3
        )
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model.eval()
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        st.error("❌ Could not load model - some features will be disabled")
        return None, None

@st.cache_data(show_spinner=False)
def predict_batch(texts, _model, _tokenizer, batch_size=32):
    """
    Batch prediction with error handling.
    
    Args:
        texts: List of texts to predict
        _model: Model with leading underscore to prevent Streamlit hashing
        _tokenizer: Tokenizer with leading underscore to prevent Streamlit hashing
        batch_size: Number of texts to process at once
        
    Returns:
        List of predictions (0, 1, or 2) for each text
    """
    if _model is None or _tokenizer is None:
        logger.warning("Model or tokenizer is None, returning default predictions")
        return [0] * len(texts)
        
    try:
        predictions = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process batch
            inputs = _tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = _model(**inputs)
                batch_preds = outputs.logits.argmax(dim=1).cpu().numpy().tolist()
                predictions.extend(batch_preds)
                
            # Update progress
            progress = (i + batch_size) / len(texts)
            st.progress(progress, text=f"Processing texts... {progress:.0%}")
            
        return predictions
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error(f"❌ Error during prediction: {str(e)}")
        return [0] * len(texts)

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Load and process data
df = load_and_preprocess_data(uploaded_file)

if df is not None:
    # Run inference
    st.info("🤖 Running cyberbullying detection model...")
    try:
        # Get texts to predict
        texts = df['cleaned_text'].astype(str).tolist()
        
        # Show prediction status
        with st.spinner("Making predictions..."):
            # Run batch prediction with unhashable arguments marked with underscore
            predictions = predict_batch(texts, _model=model, _tokenizer=tokenizer)
            
            # Add predictions to dataframe
            df['predicted_label'] = predictions
            
            # Show success message
            pred_counts = pd.Series(predictions).value_counts()
            st.success(
                f"✅ Processed {len(texts)} comments:\n"
                f"- Neutral: {pred_counts.get(0, 0)}\n"
                f"- Mild: {pred_counts.get(1, 0)}\n"
                f"- Severe: {pred_counts.get(2, 0)}"
            )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        st.error(f"❌ Error during prediction: {str(e)}")
        df['predicted_label'] = [0] * len(df)  # Safe fallback

# ===============================
# 2. Sidebar Filters and Controls
# ===============================
st.sidebar.header("🔍 Filters & Controls")

# Initialize filter variables
selected_label = 'All'
keyword = ''

# Only show data filters if we have data
if df is not None and 'label' in df.columns:
    # Label filter
    label_options = ['All'] + sorted(df['label'].unique().tolist())
    selected_label = st.sidebar.selectbox(
        'Filter by Label',
        label_options,
        help="Filter comments by their label"
    )
    
    # Keyword filter
    keyword = st.sidebar.text_input(
        'Search by Keyword',
        help="Filter comments containing this text"
    )
else:
    st.sidebar.info("Upload data to enable filtering")

# Email configuration (always show this section)
st.sidebar.markdown("---")
st.sidebar.header("📧 Email Settings")
default_email = 'alerts@example.com'
alert_email = st.sidebar.text_input('Alert recipient email', value=default_email)

# ===============================
# 3. Data Filtering
# ===============================
def filter_dataframe(df, label=None, keyword=None):
    """Filter dataframe based on label and keyword"""
    if df is None:
        return None
        
    filtered_df = df.copy()
    
    try:
        # Filter by label if specified
        if label and label != 'All' and 'label' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['label'] == label]
            
        # Filter by keyword if specified
        if keyword and 'comment_text' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['comment_text'].str.contains(
                    keyword, 
                    case=False, 
                    na=False
                )
            ]
            
        return filtered_df
    except Exception as e:
        logger.error(f"Error filtering data: {e}")
        return df  # Return original dataframe on error

# Apply filters if data is available
filtered_df = filter_dataframe(df, selected_label, keyword)

# Show filter results if filters are active
if df is not None and (selected_label != 'All' or keyword):
    total = len(df) if df is not None else 0
    filtered = len(filtered_df) if filtered_df is not None else 0
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Showing {filtered} of {total} comments")

# ===============================
# 3. Dashboard Headings
# ===============================
st.title('Cyberbullying Detection Dashboard')
st.markdown('''
This dashboard provides an overview of detected cyberbullying in social media comments, with metrics, visualizations, filtering, and real-time alerts.
''')

# Alert Configuration and Processing
def detect_alerts(df, threshold):
    """Detect potentially harmful comments based on severity threshold"""
    try:
        severe_mask = (df['predicted_label'] == 2) | (df['predicted_label'] == 'severe')
        severe_count = severe_mask.sum()
        
        alerts_triggered = len(df) > 0 and severe_count / len(df) > threshold
        flagged = df[severe_mask] if alerts_triggered else pd.DataFrame()
        
        return alerts_triggered, flagged
    except Exception as e:
        logger.error(f"Alert detection error: {e}")
        return False, pd.DataFrame()

def process_alerts(flagged_df, email_config):
    """Process detected alerts and send notifications"""
    if flagged_df.empty:
        return False

    try:
        # Prepare email content
        alert_samples = flagged_df[['comment_text', 'label']].head(10)
        alert_body = "Cyberbullying Detection Alert\n\n"
        alert_body += "The following comments were flagged as potentially harmful:\n\n"
        alert_body += '\n'.join([
            f"- {row['comment_text']} (Severity: {row['label']})" 
            for _, row in alert_samples.iterrows()
        ])
        
        # Send alert if email is configured
        if all(k in email_config for k in ['smtp_server', 'smtp_username', 'recipient_email']):
            success = send_alert_email(
                "🚨 Cyberbullying Alert: Severe Comments Detected",
                alert_body,
                email_config
            )
            if success:
                st.sidebar.success("✉️ Alert email sent successfully")
            return success
    except Exception as e:
        logger.error(f"Alert processing error: {e}")
        st.sidebar.error("❌ Failed to send alert email")
    
    return False

# Sidebar Alert Configuration
with st.sidebar.expander("Alert Settings", expanded=True):
    threshold = st.slider(
        'Alert Threshold (% severe)',
        min_value=5,
        max_value=50,
        value=15,
        step=5,
        help="Trigger alerts when severe comments exceed this percentage"
    ) / 100.0
    
    # Email Configuration
    email_config = load_email_config()
    st.write("📧 Email Alert Settings")
    new_config = {}
    new_config['smtp_server'] = st.text_input("SMTP Server", email_config.get('smtp_server', ''))
    new_config['smtp_port'] = st.number_input("SMTP Port", value=email_config.get('smtp_port', 587))
    new_config['smtp_username'] = st.text_input("SMTP Username", email_config.get('smtp_username', ''))
    new_config['smtp_password'] = st.text_input("SMTP Password", email_config.get('smtp_password', ''), type='password')
    new_config['sender_email'] = st.text_input("Sender Email", email_config.get('sender_email', ''))
    new_config['recipient_email'] = st.text_input("Recipient Email", email_config.get('recipient_email', ''))
    new_config['use_tls'] = st.checkbox("Use TLS", value=email_config.get('use_tls', True))
    
    if st.button("Save Email Settings"):
        save_email_config(new_config)
        st.success("✅ Email settings saved successfully!")

# Process Alerts
if filtered_df is not None:
    # Configure alert threshold
    threshold = st.sidebar.slider(
        'Alert Threshold (%)',
        min_value=5,
        max_value=50,
        value=15,
        step=5,
        help="Trigger alerts when severe comments exceed this percentage"
    ) / 100.0
    
    # Detect alerts
    alerts_exist, flagged_df = detect_cyberbullying_alerts(filtered_df, threshold)
    
    # Display alerts section
    st.markdown("---")
    st.header("🚨 Alert System")
    display_alerts(filtered_df, flagged_df, new_config)
    
    # Show alert stats
    if not flagged_df.empty:
        st.sidebar.markdown("---")
        st.sidebar.error(f"⚠️ {len(flagged_df)} comments flagged as severe")

# ===============================
# Data Validation Functions
# ===============================
def validate_and_clean_data(df):
    """
    Clean and validate data for metrics calculation.
    Returns validated dataframe or None if validation fails.
    """
    if df is None:
        return None
        
    try:
        # Make a copy to avoid modifying original
        valid_df = df.copy()
        
        # Check for required columns
        if 'predicted_label' not in valid_df.columns:
            logger.warning("Missing predicted_label column")
            return None
            
        if 'label' not in valid_df.columns:
            logger.warning("Missing label column")
            return None
            
        # Remove rows with missing values
        valid_df = valid_df.dropna(subset=['label', 'predicted_label'])
        
        # Convert string labels to integers if needed
        label_map = {'neutral': 0, 'mild': 1, 'severe': 2}
        
        if valid_df['label'].dtype == object:
            valid_df['label'] = valid_df['label'].map(
                lambda x: label_map.get(x, x) if isinstance(x, str) else x
            )
            
        if valid_df['predicted_label'].dtype == object:
            valid_df['predicted_label'] = valid_df['predicted_label'].map(
                lambda x: label_map.get(x, x) if isinstance(x, str) else x
            )
            
        # Convert to integers
        valid_df['label'] = valid_df['label'].astype(int)
        valid_df['predicted_label'] = valid_df['predicted_label'].astype(int)
        
        return valid_df
        
    except Exception as e:
        logger.error(f"Data validation error: {e}")
        return None

# Visualization Functions
@st.cache_data
def plot_class_distribution(label_counts):
    """Generate class distribution plot"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 6))
        label_names = {0: 'Neutral', 1: 'Mild', 2: 'Severe', 
                      'neutral': 'Neutral', 'mild': 'Mild', 'severe': 'Severe'}
        bar_labels = [label_names.get(l, str(l)) for l in label_counts.index]
        
        sns.barplot(x=bar_labels, y=label_counts.values, ax=ax)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Comment Classes')
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Distribution plot error: {e}")
        return None

@st.cache_data
def plot_confusion_matrix(y_true, y_pred, labels):
    """Generate confusion matrix heatmap"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        return fig, cm
    except Exception as e:
        logger.error(f"Confusion matrix error: {e}")
        return None, None

@st.cache_data
def generate_wordcloud(text_data, title):
    """Generate wordcloud for text data"""
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 5))
        wordcloud = WordCloud(
            width=800, 
            height=400,
            background_color='white',
            colormap='viridis'
        ).generate(' '.join(text_data))
        
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title)
        
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Wordcloud generation error: {e}")
        return None

# ===============================
# 4. Main Dashboard Content
# ===============================
st.header("📊 Analysis Dashboard")

# Display upload prompt if no data
if df is None:
    st.info("👆 Please upload a CSV file to start the analysis")
    st.markdown("""
        ### Expected CSV Format:
        Your file should contain these columns:
        - `comment_text`: The text content to analyze
        - `label`: (Optional) The true label (0: Neutral, 1: Mild, 2: Severe)
        
        Other columns will be preserved but not used for analysis.
    """)
elif filtered_df is not None:
    # Basic Metrics
    col1, col2, col3 = st.columns(3)
    
    try:
        with col1:
            total = len(filtered_df)
            delta = None
            if df is not None:
                delta = f"{total - len(df)} from filter" if total != len(df) else None
            st.metric("Total Comments", total, delta)
        
        with col2:
            if 'predicted_label' in filtered_df.columns:
                severe = len(filtered_df[filtered_df['predicted_label'] == 2])
                st.metric(
                    "Severe Comments", 
                    severe, 
                    f"{severe/total:.1%}" if total > 0 else None
                )
            else:
                st.metric("Severe Comments", "N/A")
                
        with col3:
            if 'label' in filtered_df.columns:
                labels = filtered_df['label'].unique()
                unique_labels = len(labels)
                st.metric(
                    "Label Classes",
                    unique_labels,
                    ", ".join(str(l) for l in sorted(labels)[:3]) + ("..." if len(labels) > 3 else "")
                )
            else:
                st.metric("Label Classes", "N/A")
                
    except Exception as e:
        logger.error(f"Error displaying metrics: {e}")
        st.error("❌ Error calculating metrics")

    # Class Distribution
    if 'label' in df.columns:
        st.subheader("Class Distribution")
        label_counts = df['label'].value_counts().sort_index()
        dist_fig = plot_class_distribution(label_counts)
        if dist_fig:
            st.pyplot(dist_fig)
        else:
            st.warning("⚠️ Could not generate class distribution plot")
    else:
        st.info("ℹ️ No label column found in the data")

# Confusion Matrix and Model Performance
if df is not None:
    required_cols = ['label', 'predicted_label']
    if all(col in df.columns for col in required_cols) and not df.empty:
        st.subheader("Model Performance")
        
        # Confusion Matrix
        label_names = {0: 'Neutral', 1: 'Mild', 2: 'Severe'}
        try:
            labels = [label_names.get(l, str(l)) for l in sorted(df['label'].unique())]
            cm_fig, cm = plot_confusion_matrix(df['label'], df['predicted_label'], labels)
            
            if cm_fig:
                st.pyplot(cm_fig)
                
                # Classification Report
                report = classification_report(df['label'], df['predicted_label'],
                                            target_names=labels, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                
                st.subheader("Classification Metrics")
                st.dataframe(
                    report_df.style.format("{:.3f}").background_gradient(cmap='Blues'),
                    use_container_width=True
                )
            else:
                st.warning("⚠️ Could not generate confusion matrix")
        except Exception as e:
            logger.error(f"Model performance visualization error: {e}")
            st.error("❌ Error generating model performance metrics")
    elif not set(required_cols).issubset(df.columns):
        missing = [col for col in required_cols if col not in df.columns]
        st.info(f"ℹ️ Cannot show model performance: Missing columns {', '.join(missing)}")

# Wordclouds
if df is not None and 'label' in df.columns and 'cleaned_text' in df.columns:
    st.subheader("Word Clouds by Class")
    try:
        unique_labels = sorted(df['label'].unique())
        cols = st.columns(len(unique_labels))

        for idx, (label, col) in enumerate(zip(unique_labels, cols)):
            with col:
                class_df = df[df['label'] == label]
                if not class_df.empty:
                    wc_fig = generate_wordcloud(
                        class_df['cleaned_text'], 
                        f"{label_names.get(label, f'Class {label}')} Comments"
                    )
                    if wc_fig:
                        st.pyplot(wc_fig)
                    else:
                        st.warning(f"⚠️ Could not generate wordcloud for class {label}")
    except Exception as e:
        logger.error(f"Wordcloud generation error: {e}")
        st.error("❌ Error generating word clouds")

# Sample Predictions
if df is not None:
    st.subheader("Sample Predictions")
    cols_to_show = ['comment_text']
    if 'label' in df.columns:
        cols_to_show.append('label')
    if 'predicted_label' in df.columns:
        cols_to_show.append('predicted_label')
    
    if len(cols_to_show) > 1:  # Only show if we have more than just the comment text
        try:
            sample_size = min(10, len(df))
            sample_df = df[cols_to_show].sample(sample_size)
            
            # Rename columns for display
            display_names = {
                'label': 'True Label',
                'predicted_label': 'Predicted Label'
            }
            sample_df = sample_df.rename(columns=display_names)
            
            # Format the dataframe
            style_cols = [col for col in ['True Label', 'Predicted Label'] if col in sample_df.columns]
            st.dataframe(
                sample_df.style.format({
                    'True Label': lambda x: label_names.get(x, str(x)),
                    'Predicted Label': lambda x: label_names.get(x, str(x))
                }).background_gradient(cmap='Blues', subset=style_cols),
                use_container_width=True
            )
        except Exception as e:
            logger.error(f"Sample display error: {e}")
            st.error("❌ Error displaying sample predictions")
# Wordclouds help visualize the most common words in each class, giving insight into language patterns.

# ===============================
# End of dashboard content
st.markdown('''
---
**How to use this dashboard:**
1. Upload your CSV file with comment data
2. Use the sidebar filters to analyze specific subsets
3. Configure email alerts for severe content detection
4. Review metrics and visualizations to understand the data

*Note: This dashboard is for demonstration purposes and can be extended for production use.*
''')

# ===============================
# 9. Notes
# ===============================
st.markdown('''
---
**How to use this dashboard:**
- Use the sidebar to filter comments by label or keyword.
- Review metrics and visualizations to understand model/class distribution.
- Inspect flagged comments for real-world examples.

*This dashboard is designed for demo/interview use and can be extended for production analytics.*
''')
