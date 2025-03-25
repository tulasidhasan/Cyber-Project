import cv2
import pytesseract
from PIL import Image

# Set Tesseract path manually (if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

# Example usage
if __name__ == "__main__":
    image_path = "1.jpg"  # Change this to your image file
    text = extract_text(image_path)
    print("Extracted Text:\n", text)
