import asyncio
from document_ingestor import process_document_from_url

async def main():
    url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    extracted_text = await process_document_from_url(url)
    print("📄 Extracted Text:\n")
    print(extracted_text[:1000])

if __name__ == "__main__":
    asyncio.run(main())
