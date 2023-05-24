import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from utils import get_prompt, read_pdf

os.environ["OPENAI_API_KEY"] = "OPEN_API_KEY"

# Connecction with Postgres

CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver=os.environ.get("PGVECTOR_DRIVER", "psycopg2"),
    host=os.environ.get("PGVECTOR_HOST", "localhost"),
    port=int(os.environ.get("PGVECTOR_PORT", "5432")),
    database=os.environ.get("PGVECTOR_DATABASE", "chatbot"),
    user=os.environ.get("PGVECTOR_USER", "postgres"),
    password=os.environ.get("PGVECTOR_PASSWORD", "postgres"),
)

# Read PDF

docs = read_pdf('../python_notes.pdf')

embeddings = OpenAIEmbeddings()

# Embedding and Uploading vectors on the postgres
print(f'{"*"*50} UPLOADING {"*"*50}')
db = PGVector.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="chatbot_notes",
    connection_string=CONNECTION_STRING,
    pre_delete_collection=False 
)
print(f'{"*"*50} UPLOADED {"*"*50}')