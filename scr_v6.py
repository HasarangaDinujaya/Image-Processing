import cv2
import numpy as np
import requests
import os

# Function to load image and convert to grayscale
def load_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# Function to perform multi-scale template matching
def match_template(image, template, scale_range=(0.5, 1.5), step=0.05):
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    for scale in np.arange(scale_range[0], scale_range[1], step):
        resized_template = cv2.resize(template_gray, (0, 0), fx=scale, fy=scale)
        result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
        (_, max_val, _, max_loc) = cv2.minMaxLoc(result)
        if max_val > 0.8:  # Threshold for detection
            return True, max_val, max_loc
    return False, None, None

# Dictionary for currency templates
# Each currency has multiple templates for different denominations
currency_templates = {
    'LKR': {
        '5000': ['scr/5000SCR/Bird5000.jpg']  # Add actual template image paths here
    }
}

# Function to check and detect the currency note
def detect_currency(note_image, currency_templates):
    gray_image = load_image(note_image)
    for currency, notes in currency_templates.items():
        for value, templates in notes.items():
            for template_path in templates:
                if not os.path.exists(template_path):
                    print(f"Template {template_path} not found!")
                    continue
                template = cv2.imread(template_path)
                match, max_val, loc = match_template(gray_image, template)
                if match:
                    return currency, value
    return None, None

# Function to get the exchange rate for a given currency
def get_exchange_rate(currency_code):
    try:
        response = requests.get(f'https://api.exchangerate-api.com/v4/latest/{currency_code}')
        rates = response.json()['rates']
        lkr_rate = rates.get('LKR', None)
        return lkr_rate
    except Exception as e:
        print(f"Error fetching exchange rate: {e}")
        return None

# Function to process user-uploaded image
def process_uploaded_image(user_image_path):
    currency, value = detect_currency(user_image_path, currency_templates)
    if currency and value:
        print(f"Detected: {currency} {value} note.")
        
        # Get exchange rate to LKR
        exchange_rate = get_exchange_rate(currency)
        if exchange_rate:
            converted_value = value * exchange_rate
            print(f"Exchange Rate: 1 {currency} = {exchange_rate} LKR")
            print(f"Value in LKR: {converted_value} LKR")
        else:
            print(f"Could not fetch exchange rate for {currency}.")
    else:
        print("No match found.")

# Example usage
user_image_path = 'res\original.jpg'  # Replace with actual uploaded image path
process_uploaded_image(user_image_path)
