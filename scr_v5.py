import cv2
import numpy as np
import matplotlib.pyplot as plt
from forex_python.converter import CurrencyRates

# Initialize currency templates
currency_templates = {
    'LKR': {
        '5000': ['scr/5000SCR/Bird5000.jpg']  # Add actual template image paths here
    }
}

# Function to display an image using matplotlib
def display_image(title, image, cmap_type='gray'):
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Preprocess the user input image
def preprocess_image(image_path):
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return None

    print("Converting image to grayscale...")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Display the grayscale image
    display_image("Grayscale Image", gray_image)

    print("Applying thresholding to remove noise...")
    _, thresh_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    
    # Display the thresholded image
    display_image("Thresholded Image", thresh_image)

    return thresh_image

# Perform template matching and return the best match
def match_template(image, template, scale_factors):
    print("Starting template matching...")
    best_match = None
    best_scale = None
    best_score = -1

    for scale in scale_factors:
        print(f"Resizing template by scale factor: {scale}")
        # Resize template based on scale
        resized_template = cv2.resize(template, (0, 0), fx=scale, fy=scale)

        # Perform template matching
        result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)

        # Get the best match location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > best_score:  # Choose the best matching score
            print(f"Found a better match with score: {max_val}")
            best_match = max_loc
            best_scale = scale
            best_score = max_val

        # Draw a rectangle on the matched area for visualization
        h, w = resized_template.shape
        match_image = image.copy()
        cv2.rectangle(match_image, max_loc, (max_loc[0] + w, max_loc[1] + h), 255, 2)
        
        # Display the matching result
        display_image(f'Matching Result at scale {scale}', match_image)

    return best_match, best_scale, best_score

# Align input image to the template
def align_images(image, template):
    print("Aligning images using ORB keypoint detection and BFMatcher...")
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(image, None)
    kp2, des2 = orb.detectAndCompute(template, None)

    # Use BFMatcher for matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches for visualization
    match_img = cv2.drawMatches(image, kp1, template, kp2, matches[:10], None, flags=2)
    display_image("ORB Matches", match_img, cmap_type=None)

    # Use homography for alignment
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    aligned_image = cv2.warpPerspective(image, M, (template.shape[1], template.shape[0]))

    # Show the aligned image
    display_image("Aligned Image", aligned_image)

    return aligned_image

# Detect the currency note by matching the input with templates
def detect_currency(input_image_path):
    print("Preprocessing user input image...")
    input_image = preprocess_image(input_image_path)
    if input_image is None:
        print("Error: Input image preprocessing failed.")
        return None, None

    # Iterate over each currency and denomination
    for currency, notes in currency_templates.items():
        for note, templates in notes.items():
            print(f"Checking templates for {note} {currency} note...")
            matches = 0
            total_templates = len(templates)

            for template_path in templates:
                print(f"Loading template image: {template_path}")
                template = preprocess_image(template_path)
                if template is None:
                    continue

                # Align images before matching
                print("Aligning input image with template...")
                aligned_image = align_images(input_image, template)

                # Perform template matching
                print("Matching template with aligned input image...")
                best_match, best_scale, best_score = match_template(aligned_image, template, scale_factors=[0.8, 0.9, 1.0, 1.1, 1.2])

                # Check if match score is above a threshold
                if best_score > 0.8:
                    print(f"Template matched with score: {best_score}")
                    matches += 1

            # If all templates match, we detect the note
            if matches == total_templates:
                print(f"Detected {note} of {currency}")
                return currency, note

    print("Currency note not detected.")
    return None, None

# Get exchange rate for the detected currency
def get_exchange_rate(currency_code, target_currency='LKR'):
    print(f"Getting exchange rate from {currency_code} to {target_currency}...")
    c = CurrencyRates()
    rate = c.get_rate(currency_code, target_currency)
    print(f"Exchange rate: 1 {currency_code} = {rate} {target_currency}")
    return rate

# Main function to take user input and process the detection
def main():
    input_image_path = 'res/original.jpg'  # Replace with the user's uploaded image path

    # Detect the currency and note
    print("Starting currency detection...")
    currency, note = detect_currency(input_image_path)

    if currency and note:
        print(f"{note} note detected from {currency}")
        exchange_rate = get_exchange_rate(currency)
        print(f"{note} from {currency}, Exchange Rate with LKR: {exchange_rate}")
    else:
        print("Could not detect the currency note.")

if __name__ == "__main__":
    main()