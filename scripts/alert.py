import csv
from utils.notifier import send_email

def log_alert(message, alert_file='logs/alert_log.csv'):
    with open(alert_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([message])


def check_for_alerts(df, threshold=0.15):
    """
    Check for cyberbullying alerts based on predicted_label column.
    Returns flagged comments if threshold is exceeded, else empty list.
    """
    if 'predicted_label' not in df.columns:
        return []
    severe_mask = (df['predicted_label'] == 2) | (df['predicted_label'] == 'severe')
    severe_count = severe_mask.sum()
    total = len(df)
    if total == 0:
        return []
    if severe_count / total > threshold:
        flagged = df[severe_mask]
        log_alert(f'Cyberbullying detected! {severe_count}/{total} severe')
        return flagged
    return []
