import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from utils import get_prompt

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

embeddings = OpenAIEmbeddings()

store = PGVector(
    connection_string=CONNECTION_STRING, 
    embedding_function=embeddings,
    collection_name="chatbot_notes",
)


def get_answer(question):
    PROMPT = get_prompt(question)
    # Retrive vectors from postgres
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=store.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})
    answer = qa({"query": question})
    return answer


def main():
    name = input("Please enter your name: ")
    while True:
        question = input(f"Hello {name}: Ask me anything or type quit to exit: ")
        if question == "quit":
            break
        answer = get_answer(question)
        print(answer)
        print(f'{"*"*50} ANSWER {"*"*50}')
        print(f"Answer: {answer['result']}")

if __name__ == "__main__":
    main()
