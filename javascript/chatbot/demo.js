import { readPdf, uploadDocs } from "./utils.js";
import { createClient } from "redis";
import { Document } from "langchain/document";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { RetrievalQAChain, loadQAStuffChain } from "langchain/chains";
import { RedisVectorStore } from "langchain/vectorstores/redis";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { PromptTemplate } from "langchain/prompts";

async function splitAndUploadDocs() {
  const docs = await readPdf("python_notes.pdf");

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 1,
  });

  const docOutput = await splitter.createDocuments([docs[0].pageContent]);

  try {
    uploadDocs(docOutput);
  } catch (exceptionVar) {
    console.log("Some error occurred");
    console.log(exceptionVar);
  }

  console.log("*******************UPLOADED*******************");
}

async function performQuestionAnswering() {
  const QA_PROMPT =
    PromptTemplate.fromTemplate(`Use the following pieces of context to answer the question at the end. If the exact question is asked and the answer is in the documents, provide the answer. Otherwise, if you don't know the answer or it cannot be found in the documents, just say, "I don't know the answer."
  {context}
  Question: {question}
  Answer:`);

  // console.log(QA_PROMPT);

  const model = new ChatOpenAI({
    openAIApiKey: "OPEN_API_KEY",
  });

  const client = createClient({
    url: process.env.REDIS_URL ?? "redis://localhost:6379",
  });
  await client.connect();

  const vectorStore = new RedisVectorStore(
    new OpenAIEmbeddings({
      openAIApiKey: "OPEN_API_KEY",
    }),
    {
      redisClient: client,
      indexName: "docs",
    }
  );

  const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever(), {
    prompt: QA_PROMPT,
  });

  const res = await chain.call({
    query: "What is Car?",
  });

  console.log({ res });
}

async function main() {
  await splitAndUploadDocs();
  await performQuestionAnswering();
}

main().catch((error) => {
  console.error("An error occurred:", error);
});
