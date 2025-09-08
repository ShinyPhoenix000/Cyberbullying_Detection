import csv
import datetime
import os

def log_alert(message, alert_file='logs/alert_log.csv'):
    os.makedirs(os.path.dirname(alert_file), exist_ok=True)
    with open(alert_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now().isoformat(), message])

def log_event(message, log_file='logs/log.csv'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now().isoformat(), message])

# Optional: Email/Slack notification placeholder
def send_notification(message):
    pass  # Implement email/Slack logic here
