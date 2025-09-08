
import smtplib
import ssl
from email.message import EmailMessage

# --- SMTP CONFIGURATION ---
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 465
SENDER_EMAIL = 'your_sender_email@gmail.com'  # Replace with your sender email
APP_PASSWORD = 'your_app_specific_password'   # Replace with your app password (not your main Gmail password)

def send_email(subject, message, to_email):
    """
    Send an email using Gmail SMTP with SSL.
    Args:
        subject (str): Email subject
        message (str): Email body
        to_email (str): Recipient email address
    """
    email = EmailMessage()
    email['From'] = SENDER_EMAIL
    email['To'] = to_email
    email['Subject'] = subject
    email.set_content(message)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as smtp:
        smtp.login(SENDER_EMAIL, APP_PASSWORD)
        smtp.send_message(email)
