import aiohttp
import os
import tempfile
import urllib.parse
import textract
import mimetypes

async def download_file(url):
    print(f"📥 Downloading file from: {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise Exception(f"Failed to download file. HTTP Status: {response.status}")
            
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(await response.read())
            temp_file.close()
            
            file_name = os.path.basename(urllib.parse.urlparse(url).path)
            file_ext = os.path.splitext(file_name)[1].lower()
            print(f"📄 File saved as: {temp_file.name} (type: {file_ext})")
            return temp_file.name, file_ext

def extract_text(file_path, file_ext):
    if file_ext in [".pdf", ".docx", ".txt"]:
        try:
            text = textract.process(file_path).decode("utf-8")
            return text
        except Exception as e:
            raise Exception(f"Error extracting text: {str(e)}")
    else:
        raise ValueError("❌ Unsupported file type. Please use PDF, DOCX, or TXT.")

async def process_document_from_url(url):
    file_path, file_ext = await download_file(url)
    text = extract_text(file_path, file_ext)
    os.remove(file_path)
    return text
