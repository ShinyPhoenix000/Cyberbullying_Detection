"""
Production-ready training script for Cyberbullying Detection using DistilBERT.
Fine-tunes the model and saves it for use in the Streamlit dashboard.
"""
import os
import sys
import logging
import time
from pathlib import Path
import pandas as pd
import torch
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    __version__ as transformers_version
)
from packaging import version
import string
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Check transformers version for feature compatibility
TRANSFORMERS_MIN_VERSION = "4.18.0"
HAS_STRATEGY_ARGS = version.parse(transformers_version) >= version.parse(TRANSFORMERS_MIN_VERSION)
logger = logging.getLogger(__name__)

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
def suppress_warnings():
    """Suppress warnings for clean output during demo/interview."""
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
    warnings.filterwarnings('ignore')

suppress_warnings()

# Download NLTK stopwords if not already present
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# 2. Load CSV file (expects columns: 'comment', 'label')
csv_path = os.path.join(os.path.dirname(__file__), '../data/twitter_data.csv')
df = pd.read_csv(csv_path)

# 3. Preprocess the 'comment' text: lowercase, remove punctuation, remove stopwords, strip spaces
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    tokens = [word for word in text.split() if word not in stop_words]
    # Join and strip extra spaces
    return ' '.join(tokens).strip()

# 4. Rename 'comment' to 'comment_text' and strip column names
df.columns = [c.strip() for c in df.columns]
if 'comment' in df.columns:
    df.rename(columns={'comment': 'comment_text'}, inplace=True)

# Show detected columns and sample preprocessed text
print('Detected columns:', list(df.columns))
df = df[['comment_text', 'label']].dropna().sample(n=200, random_state=42)  # 12. Use small subset for speed
df['cleaned_text'] = df['comment_text'].astype(str).apply(clean_text)
print('Sample preprocessed text:', df['cleaned_text'].head(3).tolist())

# 5. Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
)

# 6. Tokenize using HuggingFace AutoTokenizer (truncation/padding)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)

# 7. PyTorch Dataset class
class CyberbullyingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 9. Ensure labels are torch.long for CrossEntropyLoss (required for multi-class)
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = CyberbullyingDataset(train_encodings, y_train)
test_dataset = CyberbullyingDataset(test_encodings, y_test)

# 8. Load DistilBERT model for sequence classification (3 classes)
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# 9. CrossEntropyLoss is used for multi-class classification, so labels must be torch.long (see above)
#    This is why we set dtype=torch.long in the Dataset class.

# Check if fine-tuned model exists
model_save_path = Path("models/distilbert_cyberbullying")
if model_save_path.exists():
    logger.info(f"Loading existing fine-tuned model from {model_save_path}")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
        tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        logger.info("Successfully loaded fine-tuned model")
    except Exception as e:
        logger.error(f"Error loading fine-tuned model: {e}")
        logger.info("Training new model instead")
        model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
else:
    logger.info("No fine-tuned model found. Training new model...")
    # Set up training arguments based on transformers version
    base_args = {
        'output_dir': str(model_save_path),
        'num_train_epochs': 3,
        'per_device_train_batch_size': 16,
        'per_device_eval_batch_size': 16,
        'logging_dir': 'logs',
        'logging_steps': 10,
        'report_to': 'none',  # Disable wandb/tensorboard
    }
    
    if HAS_STRATEGY_ARGS:
        logger.info("Using newer transformers with strategy arguments")
        advanced_args = {
            'evaluation_strategy': 'epoch',
            'save_strategy': 'epoch',
            'save_total_limit': 2,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'f1',
            'greater_is_better': True,
        }
        training_args = TrainingArguments(**base_args, **advanced_args)
    else:
        logger.info("Using older transformers without strategy arguments")
        training_args = TrainingArguments(**base_args)

    # Trainer setup with evaluation metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc, 'f1': f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    logger.info("Starting training...")
    try:
        # Run training
        train_result = trainer.train()
        logger.info(f"Training completed. Metrics: {train_result.metrics}")
        
        # Create save directory if it doesn't exist
        os.makedirs(str(model_save_path), exist_ok=True)
        
        # Save the final fine-tuned model
        logger.info(f"Saving fine-tuned model to {model_save_path}")
        model.save_pretrained(str(model_save_path))
        tokenizer.save_pretrained(str(model_save_path))
        
        # Save training metrics and config
        metrics_file = model_save_path / "training_metrics.txt"
        logger.info(f"Saving training metrics to {metrics_file}")
        with open(metrics_file, "w") as f:
            f.write(f"Training Metrics:\n{train_result.metrics}\n")
            f.write(f"\nModel Configuration:\n{model.config}")
        
        # Validate saved files
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json', 'special_tokens_map.json']
        missing_files = [f for f in required_files if not (model_save_path / f).exists()]
        if missing_files:
            logger.warning(f"Some model files are missing: {missing_files}")
        else:
            logger.info("Successfully saved all required model files!")
            
    except Exception as e:
        logger.error(f"Training or saving failed: {e}")
        sys.exit(1)

# Evaluate on test set
logger.info("Evaluating model on test set...")
preds = trainer.predict(test_dataset)
y_pred = preds.predictions.argmax(axis=1)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1-score:', f1_score(y_test, y_pred, average='weighted'))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# 17. Visualizations (for interview/demo insight)
# --- Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[-1, 0, 1], yticklabels=[-1, 0, 1])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Heatmap')
plt.tight_layout()
plt.show()
# This plot helps visually assess where the model is making correct vs. incorrect predictions.

# --- Class Distribution Bar Charts ---
import numpy as np
true_counts = np.bincount([int(l) for l in y_test], minlength=3)
pred_counts = np.bincount([int(l) for l in y_pred], minlength=3)
labels = [-1, 0, 1]
bar_width = 0.35
x = np.arange(len(labels))
plt.figure(figsize=(7,5))
plt.bar(x - bar_width/2, true_counts, width=bar_width, label='True', color='skyblue')
plt.bar(x + bar_width/2, pred_counts, width=bar_width, label='Predicted', color='salmon')
plt.xticks(x, labels)
plt.xlabel('Class Label')
plt.ylabel('Count')
plt.title('Class Distribution: True vs Predicted')
plt.legend()
plt.tight_layout()
plt.show()
# These bar charts help you quickly see if the model is biased toward certain classes or missing others.

# Optional: SHAP Explainability
def generate_shap_explanations(model, tokenizer, texts, max_samples=5):
    try:
        import shap
        logger.info("Generating SHAP explanations...")
        explainer = shap.Explainer(model, masker=tokenizer)
        shap_values = explainer(texts[:max_samples])
        shap.plots.text(shap_values, save=True)
        logger.info("SHAP explanations saved")
    except Exception as e:
        logger.warning(f"Could not generate SHAP explanations: {e}")

def validate_model_files():
    """Validate that model files exist and can be loaded."""
    model_path = Path("models/distilbert_cyberbullying")
    required_files = ['config.json', 'pytorch_model.bin', 'tokenizer_config.json', 'vocab.txt']
    
    if not model_path.exists():
        return False
        
    missing_files = [f for f in required_files if not (model_path / f).exists()]
    if missing_files:
        logger.warning(f"Missing required model files: {missing_files}")
        return False
    return True

if __name__ == '__main__':
    try:
        start_time = time.time()
        logger.info("Starting cyberbullying detection model training pipeline")
        
        # Run main training/evaluation logic
        if not validate_model_files():
            logger.info("Training new model as valid model files not found")
        
        # Optional: Generate SHAP explanations
        if '--explain' in sys.argv:
            generate_shap_explanations(model, tokenizer, X_test)
        
        duration = time.time() - start_time
        logger.info(f"Pipeline completed in {duration:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)