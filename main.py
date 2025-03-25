import os
from image_to_text import extract_text as extract_text_from_image
from pdf_to_text import extract_text_from_pdf
from crime_classifier import classify_crime

def analyze_crime(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found!")
        return

    # Determine file type
    ext = os.path.splitext(file_path)[-1].lower()

    if ext in [".png", ".jpg", ".jpeg"]:
        text = extract_text_from_image(file_path)
    elif ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    else:
        print("Unsupported file type!")
        return

    print("\nExtracted Text:\n", text)

    if text.strip():
        crime_type = classify_crime(text)
        print("\nPredicted Crime:", crime_type["label"])
    else:
        print("No text found! Check the image quality.")

if __name__ == "__main__":
    file_path = "1.jpg"  # Ensure this file exists in the correct directory
    analyze_crime(file_path)
