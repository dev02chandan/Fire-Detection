import cv2
import numpy as np

# Fire Detection Functions
def detect_fire(image, lower_fire=np.array([15, 100, 100]), upper_fire=np.array([35, 255, 255])):
    """
    Detect fire in the given image using HSV color thresholding and morphological operations.
    
    Parameters:
    - image: numpy array, the image to process
    - lower_fire: np.array, lower HSV threshold for fire-like colors
    - upper_fire: np.array, upper HSV threshold for fire-like colors
    
    Returns:
    - fire_mask_cleaned: np.array, binary mask showing fire regions
    """
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask based on the color range
    fire_mask = cv2.inRange(hsv_image, lower_fire, upper_fire)
    
    # Perform morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    fire_mask_cleaned = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask_cleaned = cv2.morphologyEx(fire_mask_cleaned, cv2.MORPH_CLOSE, kernel)

    return fire_mask_cleaned

def classify_fire_presence(fire_mask, fire_threshold=0.01):
    """
    Classify whether fire is present based on the percentage of fire pixels in the image.
    
    Parameters:
    - fire_mask: np.array, binary mask where fire regions are white (255)
    - fire_threshold: float, the percentage threshold above which we classify the image as containing fire (default 1%)
    
    Returns:
    - bool: True if fire is detected, False otherwise
    """
    
    # Calculate the number of fire pixels (non-zero values in the fire mask)
    fire_pixel_count = np.count_nonzero(fire_mask)
    
    # Calculate the total number of pixels in the image
    total_pixel_count = fire_mask.size
    
    # Calculate the fire pixel percentage
    fire_percentage = fire_pixel_count / total_pixel_count
    
    # Compare with the threshold
    if fire_percentage >= fire_threshold:
        return True  # Fire detected
    else:
        return False  # No fire detected