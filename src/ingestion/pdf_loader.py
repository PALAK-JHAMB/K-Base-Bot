# import os
# import yaml
# from unstructured.partition.pdf import partition_pdf
# from unstructured.documents.elements import Table, Title, Text
# from langchain.docstore.document import Document
# import base64
# from google.generativeai.types import HarmCategory, HarmBlockThreshold
# import google.generativeai as genai
# import streamlit as st # Import Streamlit

# # --- Placeholder for Gemini Vision Functionality ---
# # This function will describe an image using the Gemini Pro Vision model.
# def get_image_description(image_bytes: bytes) -> str:
#     """Uses Gemini Pro Vision to describe an image."""
    
#     # Get the API key from Streamlit secrets instead of hardcoding it
#     # This assumes you have GEMINI_API_KEY set in your secrets.toml file
#     try:
#         api_key = st.secrets["API_KEY"]
#         genai.configure(api_key=api_key)
#     except KeyError:
#         return "[Image Description: Error - GEMINI_API_KEY not found in Streamlit secrets.]"
#     except Exception as e:
#         return f"[Image Description: Error configuring Gemini - {e}]"


#     image_parts = [{"mime_type": "image/jpeg", "data": image_bytes}]
#     prompt_parts = [
#         "Describe this image from a user manual in detail. Focus on any text, buttons, or interface elements shown. What is the user meant to do here?\n",
#         *image_parts
#     ]
    
#     try:
#         model = genai.GenerativeModel('gemini-pro-vision')
#         response = model.generate_content(
#             prompt_parts,
#             # Block potentially sensitive content for safety
#             safety_settings={
#                 HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#                 HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
#             }
#         )
#         return f"[Image Description: {response.text}]"
#     except Exception as e:
#         return f"[Image Description: Error processing image - {e}]"


# def load_and_process_pdfs(pdf_folder_path: str, config: dict) -> list[Document]:
#     """
#     Loads and processes PDFs using the 'unstructured' library, handling text and tables.
#     Optionally processes images using a multimodal model.
#     """
#     documents = []
#     ingestion_config = config.get('ingestion', {})
    
#     # The API key is now managed by Streamlit secrets, not passed via config
#     # api_key = gemini_config.get('api_key') 
    
#     strategy = ingestion_config.get('parsing_strategy', 'fast')
#     process_images_flag = ingestion_config.get('process_images', False)

#     for file in os.listdir(pdf_folder_path):
#         if not file.endswith('.pdf'):
#             continue
            
#         pdf_path = os.path.join(pdf_folder_path, file)
#         print(f"Processing {pdf_path} with strategy '{strategy}'...")
        
#         # Use unstructured to partition the PDF
#         elements = partition_pdf(
#             filename=pdf_path,
#             strategy=strategy,
#             infer_table_structure=True, # Important for table quality
#             extract_images_in_pdf=process_images_flag, # Only extract images if flag is True
#         )
        
#         page_content = ""
#         for element in elements:
#             if isinstance(element, Table):
#                 # Format tables clearly for the LLM
#                 page_content += "\n\n--- TABLE START ---\n"
#                 page_content += element.text
#                 page_content += "\n--- TABLE END ---\n\n"
#             elif isinstance(element, Title):
#                 page_content += f"\n## {element.text}\n\n"
#             elif isinstance(element, Text):
#                 page_content += element.text + "\n"
#             # This requires 'unstructured' with image extraction capabilities
#             elif process_images_flag and type(element).__name__ == 'Image':
#                 print(f"  - Describing image on page {element.metadata.page_number}...")
#                 # This function now uses the key from secrets directly
#                 image_description = get_image_description(element.image_bytes)
#                 page_content += image_description + "\n"

#         if page_content:
#             documents.append(Document(
#                 page_content=page_content,
#                 metadata={'source': file}
#             ))
            
#     return documents
# src/ingestion/pdf_loader.py
import os
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Title, Text
from langchain.docstore.document import Document
import base64
import streamlit as st
from openai import OpenAI # Use the OpenAI library

def get_image_description(image_bytes: bytes) -> str:
    """Uses OpenAI's GPT-4o model to describe an image."""
    try:
        # Get the OpenAI API key from secrets
        api_key = st.secrets["OPENAI_API_KEY"] # Use the correct secret name
        client = OpenAI(api_key=api_key)
    except KeyError:
        return "[Image Description: Error - OPENAI_API_KEY not found in Streamlit secrets.]"
    except Exception as e:
        return f"[Image Description: Error configuring OpenAI - {e}]"

    try:
        # Encode the image bytes to a base64 string
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        response = client.chat.completions.create(
            model="gpt-4o", # Or "gpt-4-turbo"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image from a user manual in detail. Focus on any text, buttons, or interface elements shown. What is the user meant to do here?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )
        description = response.choices[0].message.content
        return f"[Image Description: {description}]"
    except Exception as e:
        return f"[Image Description: Error processing image with OpenAI - {e}]"

def load_and_process_pdfs(pdf_folder_path: str, config: dict) -> list[Document]:
    # ... [The rest of this function remains exactly the same as your original] ...
    # It will now call the new OpenAI-powered get_image_description function.
    documents = []
    ingestion_config = config.get('ingestion', {})
    strategy = ingestion_config.get('parsing_strategy', 'fast')
    process_images_flag = ingestion_config.get('process_images', False)

    for file in os.listdir(pdf_folder_path):
        if not file.endswith('.pdf'):
            continue
        
        pdf_path = os.path.join(pdf_folder_path, file)
        print(f"Processing {pdf_path} with strategy '{strategy}'...")
        
        elements = partition_pdf(
            filename=pdf_path,
            strategy=strategy,
            infer_table_structure=True,
            extract_images_in_pdf=process_images_flag,
        )
        
        page_content = ""
        for element in elements:
            if isinstance(element, Table):
                page_content += f"\n\n--- TABLE START ---\n{element.text}\n--- TABLE END ---\n\n"
            elif isinstance(element, Title):
                page_content += f"\n## {element.text}\n\n"
            elif isinstance(element, Text):
                page_content += element.text + "\n"
            elif process_images_flag and type(element).__name__ == 'Image':
                print(f"  - Describing image on page {element.metadata.page_number}...")
                image_description = get_image_description(element.image_bytes)
                page_content += image_description + "\n"

        if page_content:
            documents.append(Document(
                page_content=page_content,
                metadata={'source': file}
            ))
            
    return documents