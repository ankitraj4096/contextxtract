

import textract
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

async def process_document_from_file(file_path):
    try:
        print(f"📄 Processing file: {file_path}")
        
        # Extract raw text from PDF
        text = textract.process(file_path).decode("utf-8")

        # Remove all whitespace (spaces, newlines, tabs, etc.)
        cleaned_text = ''.join(text.split())

        # Save the cleaned text to a .txt file
        output_path = "output_cleaned.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print(f"✅ Cleaned text saved to {output_path}")
        return cleaned_text

    except Exception as e:
        raise RuntimeError(f"❌ Error extracting text: {e}")
