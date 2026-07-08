from pypdf import PdfReader # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter #type:ignore


# Reading the entire text from the PDF -----------------------------------------------------------

def read_pdf(file_path:str)->str:
    reader=PdfReader(file_path)

    text=""

    for page in reader.pages:
        page_text=page.extract_text()       
        if page_text:
            text += page_text + "\n"

    return text



def create_chunks(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    return chunks








                     
