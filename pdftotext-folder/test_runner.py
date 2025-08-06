import asyncio
from hackrx_main_simple import process_document_from_file

async def main():
    file_path = (r"C:\Users\Vans\Music\hackrx-llm-system\contextxtract\pdftotext-folder\Arogya Sanjeevani Policy.pdf")

    await process_document_from_file(file_path)

asyncio.run(main())
