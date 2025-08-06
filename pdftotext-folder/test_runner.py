
import asyncio
from hackrx_main_simple import process_document_from_file

async def main():
    file_path = "C:/Users/Vans/Music/hackrx-llm-system/Arogya Sanjeevani Policy.pdf"
    await process_document_from_file(file_path)

if __name__ == "__main__":
    asyncio.run(main())
