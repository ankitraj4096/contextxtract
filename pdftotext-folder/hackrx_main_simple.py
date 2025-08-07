

# import textract
# from dotenv import load_dotenv
# import os

# # Load environment variables from .env file
# load_dotenv()

# async def process_document_from_file(file_path):
#     try:
#         print(f"📄 Processing file: {file_path}")
        
#         # Extract raw text from PDF
#         text = textract.process(file_path).decode("utf-8")

#         # Remove all whitespace (spaces, newlines, tabs, etc.)
#         cleaned_text = ''.join(text.split())

#         # Save the cleaned text to a .txt file
#         output_path = "output_cleaned.txt"
#         with open(output_path, "w", encoding="utf-8") as f:
#             f.write(cleaned_text)

#         print(f"✅ Cleaned text saved to {output_path}")
#         return cleaned_text

#     except Exception as e:
#         raise RuntimeError(f"❌ Error extracting text: {e}")


# import pdfplumber
# import os

# def extract_text_with_spaces(pdf_path, output_txt_path=None):
#     all_text = ""
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             text = page.extract_text()
#             if text:
#                 all_text += text + "\n"

#     if output_txt_path:
#         with open(output_txt_path, "w", encoding="utf-8") as f:
#             f.write(all_text)

#     return all_text


# # Example usage:
# if __name__ == "__main__":
#     input_pdf = "Arogya Sanjeevani Policy.pdf"
#     output_txt = "output_with_spaces.txt"
    
#     text = extract_text_with_spaces(input_pdf, output_txt)
#     print("✅ Text extracted with proper spacing and saved to:", output_txt)

import textract
import os

async def process_document_from_file(file_path):
    print(f"📄 Processing file: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File does not exist: {file_path}")

    try:
        # Extract text from the PDF
        text = textract.process(file_path).decode("utf-8")

        # Optional: Clean and preserve spacing (remove weird spacing issues)
        cleaned_text = ' '.join(text.split())

        # Save to .txt file (same name as the PDF, but .txt extension)
        output_path = os.path.splitext(file_path)[0] + ".txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print(f"✅ Text extracted and saved to: {output_path}")
        return cleaned_text

    except Exception as e:
        raise RuntimeError(f"❌ Error extracting text: {e}")
