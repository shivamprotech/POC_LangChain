from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader


def read_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(pages)
    return texts


def read_text_file(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


def get_prompt(question):
    template = """Use the following pieces of context to answer the question at the end. If the exact question is asked then only answer otherwise if you don't know the answer or exact question is not in the pieces of context, just say that you don't know, don't try to make up an answer.
    {context}

    Question: {question}
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )
    print(f'{"*"*50} PROMPT {"*"*50}')
    print(PROMPT)
    return PROMPT