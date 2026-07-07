
from langchain.text_splitter import RecursiveCharacterTextSplitter #type:ignore

def create_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    return chunks







# Reading the entire text from the PDF -----------------------------------------------------------
# from pypdf import PdfReader # type: ignore

# def read_pdf(file_path:str)->str:
#     reader=PdfReader(file_path)

#     text=""

#     for page in reader.pages:
        # page_text=page.extract_text()       
#         if page_text:
#             text += page_text + "\n"

#     return text
                     
