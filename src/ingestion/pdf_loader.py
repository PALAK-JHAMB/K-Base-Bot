import os
import yaml
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Title, Text
from langchain.docstore.document import Document
import base64
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import google.generativeai as genai

# --- Placeholder for Gemini Vision Functionality ---
# This function will describe an image using the Gemini Pro Vision model.
# Note: This is a more advanced step and may incur costs.
def get_image_description(image_bytes: bytes, api_key: str) -> str:
    """Uses Gemini Pro Vision to describe an image."""
    genai.configure(api_key="AIzaSyDnlZ4tLCQRwK73vi0MLKhBsg2_grYP9f8")
    
    image_parts = [{"mime_type": "image/jpeg", "data": image_bytes}]
    prompt_parts = [
        "Describe this image from a user manual in detail. Focus on any text, buttons, or interface elements shown. What is the user meant to do here?\n",
        *image_parts
    ]
    
    try:
        model = genai.GenerativeModel('gemini-pro-vision')
        response = model.generate_content(
            prompt_parts,
            # Block potentially sensitive content for safety
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }
        )
        return f"[Image Description: {response.text}]"
    except Exception as e:
        return f"[Image Description: Error processing image - {e}]"


def load_and_process_pdfs(pdf_folder_path: str, config: dict) -> list[Document]:
    """
    Loads and processes PDFs using the 'unstructured' library, handling text and tables.
    Optionally processes images using a multimodal model.
    """
    documents = []
    gemini_config = config.get('gemini', {})
    ingestion_config = config.get('ingestion', {})
    
    api_key = gemini_config.get('api_key')
    strategy = ingestion_config.get('parsing_strategy', 'fast')
    process_images_flag = ingestion_config.get('process_images', False)

    for file in os.listdir(pdf_folder_path):
        if not file.endswith('.pdf'):
            continue
            
        pdf_path = os.path.join(pdf_folder_path, file)
        print(f"Processing {pdf_path} with strategy '{strategy}'...")
        
        # Use unstructured to partition the PDF
        elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
            infer_table_structure=True, # Important for table quality
            extract_images_in_pdf=process_images_flag, # Only extract images if flag is True
        )
        
        page_content = ""
        for element in elements:
            if isinstance(element, Table):
                # Format tables clearly for the LLM
                page_content += "\n\n--- TABLE START ---\n"
                page_content += element.text
                page_content += "\n--- TABLE END ---\n\n"
            elif isinstance(element, Title):
                page_content += f"\n## {element.text}\n\n"
            elif isinstance(element, Text):
                page_content += element.text + "\n"
            # This requires 'unstructured' with image extraction capabilities
            elif process_images_flag and type(element).__name__ == 'Image' and api_key:
                print(f"  - Describing image on page {element.metadata.page_number}...")
                image_description = get_image_description(element.image_bytes, api_key)
                page_content += image_description + "\n"

        if page_content:
            documents.append(Document(
                page_content=page_content,
                metadata={'source': file}
            ))
            
    return documents