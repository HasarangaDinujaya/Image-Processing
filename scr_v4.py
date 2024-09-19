import cv2
import numpy as np
from forex_python.converter import CurrencyRates

# Initialize currency templates
currency_templates = {
    'LKR': {
        '5000_ruppies': ['scr/5000SCR/AutoBirdThresh5000.png', 'scr/5000SCR/AutoLogoThresh5000.png', 'scr/5000SCR/AutoValueThresh5000.png'] # Add actual template image paths here

    }
    # 'EUR': {
    #     '5_euro': ['template1.jpg', 'template2.jpg', 'template3.jpg']
    # }
    # Add more currencies and denominations here
}

# Preprocess the user input image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to remove noise
    _, thresh_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    return thresh_image

# Perform template matching and return the best match
def match_template(image, template, scale_factors):
    best_match = None
    best_scale = None
    best_score = -1

    for scale in scale_factors:
        # Resize template based on scale
        resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)

        # Perform template matching
        result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)

        # Get the best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:  # Choose the best matching score
            best_match = max_loc
            best_scale = scale
            best_score = max_val

    return best_match, best_scale, best_score

# Align input image to the template
def align_images(image, template):
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(image, None)
    kp2, des2 = orb.detectAndCompute(template, None)

    # Use BFMatcher for matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Use homography for alignment
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned_image = cv2.warpPerspective(image, M, (template.shape[1], template.shape[0]))

    return aligned_image

# Detect the currency note by matching the input with templates
def detect_currency(input_image_path):
    # Load and preprocess user input
    input_image = preprocess_image(input_image_path)

    # Iterate over each currency and denomination
    for currency, notes in currency_templates.items():
        for note, templates in notes.items():
            matches = 0
            total_templates = len(templates)

            for template_path in templates:
                template = preprocess_image(template_path)

                # Align images before matching
                aligned_image = align_images(input_image, template)

                # Perform template matching
                best_match, best_scale, best_score = match_template(aligned_image, template, scale_factors=[0.8, 0.9, 1.0, 1.1, 1.2])

                # Check if match score is above a threshold
                if best_score > 0.8:
                    matches += 1

            # If all templates match, we detect the note
            if matches == total_templates:
                print(f"Detected {note} of {currency}")
                return currency, note

    print("Currency note not detected.")
    return None, None

# Get exchange rate for the detected currency
def get_exchange_rate(currency_code, target_currency='LKR'):
    c = CurrencyRates()
    rate = c.get_rate(currency_code, target_currency)
    return rate

# Main function to take user input and process the detection
def main():
    input_image_path = 'res/original.jpg' # Replace with the user's uploaded image path

    # Detect the currency and note
    currency, note = detect_currency(input_image_path)

    if currency and note:
        exchange_rate = get_exchange_rate(currency)
        print(f"{note} from {currency}, Exchange Rate with LKR: {exchange_rate}")
    else:
        print("Could not detect the currency note.")

if __name__ == "__main__":
    main()
