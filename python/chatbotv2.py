import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from utils import get_prompt
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import QA_PROMPT


os.environ["OPENAI_API_KEY"] = "OPEN_API_KEY"
count = 0
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
    print("Inside")
    PROMPT = get_prompt(question)
    # Retrive vectors from postgres
    chat = []
    memory = ConversationBufferMemory(memory_key="chat", return_messages=True)
    # llm = ChatOpenAI()
    # question_generator = LLMChain(llm=llm, prompt=PROMPT)

    # doc_chain = load_qa_chain(llm, chain_type="stuff")

    # qa = ConversationalRetrievalChain(
    #     retriever=store.as_retriever(),
    #     return_source_documents=True,
    #     question_generator=question_generator,
    #     combine_docs_chain=doc_chain,
    #     # memory=memory
    # )
    query = question
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(), store.as_retriever(), memory=memory, condense_question_prompt=PROMPT)
    result = qa({"question": query, "chat_history": chat})
    return result


def main():
    name = input("Please enter your name: ")
    while True:
        question_input = input(f"Hello {name}: Ask me anything or type quit to exit: ")
        if question_input.lower() == "quit":
            break
        answer = get_answer(question_input)
        print(answer)
        print(f'{"*"*50} ANSWER {"*"*50}')
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()