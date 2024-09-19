import cv2
import numpy as np
import requests
import os
import matplotlib.pyplot as plt

# Function to load image and convert to grayscale
def load_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    return image, gray

# Function to perform multi-scale template matching and visualize intermediate steps
def match_template(image, template, scale_range=(0.5, 1.5), step=0.05):
    # Ensure both image and template are grayscale
    if len(template.shape) > 2:  # If the template is not grayscale
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template
    
    # Loop through the scale range and perform template matching
    for scale in np.arange(scale_range[0], scale_range[1], step):
        resized_template = cv2.resize(template_gray, (0, 0), fx=scale, fy=scale)
        if image.shape[0] < resized_template.shape[0] or image.shape[1] < resized_template.shape[1]:
            # Skip the scaling if the template becomes larger than the input image
            continue
        result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
        (_, max_val, _, max_loc) = cv2.minMaxLoc(result)
        
        # Visualize the template and the result at this scale
        plt.figure(figsize=(10, 5))
        
        # Plot the original image
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Plot the template being used at this scale
        plt.subplot(1, 3, 2)
        plt.title(f"Template (Scale {scale:.2f})")
        plt.imshow(resized_template, cmap='gray')

        # Plot the result of the template matching
        plt.subplot(1, 3, 3)
        plt.title("Template Matching Result")
        plt.imshow(result, cmap='hot')
        plt.colorbar()

        plt.show()

        # If a good match is found, return it
        if max_val > 0.8:  # Threshold for detection
            return True, max_val, max_loc, scale, resized_template
    return False, None, None, None, None

# Dictionary for currency templates
currency_templates = {
    'LKR': {
        '5000': ['scr/5000SCR/Bird5000.jpg']  # Add actual template image paths here
    },
    'USD': {
        'dollor one': ['scr\OneDollorSCR\OneDollor.jpg']
    }
    # Add more currency templates here
}
# Function to detect currency note
def detect_currency(note_image, currency_templates):
    original_image, gray_image = load_image(note_image)
    for currency, notes in currency_templates.items():
        for value, templates in notes.items():
            for template_path in templates:
                if not os.path.exists(template_path):
                    print(f"Template {template_path} not found!")
                    continue
                template = cv2.imread(template_path)
                match, max_val, loc, scale, resized_template = match_template(gray_image, template)
                if match:
                    # Draw rectangle on the detected region
                    h, w = resized_template.shape[:2]
                    top_left = loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    detected_img = original_image.copy()
                    cv2.rectangle(detected_img, top_left, bottom_right, (0, 255, 0), 3)
                    
                    # Plot the result
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.title("Detected Currency Region")
                    plt.imshow(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))

                    plt.subplot(1, 2, 2)
                    plt.title("Template Used for Matching")
                    plt.imshow(resized_template, cmap='gray')

                    plt.show()
                    
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
