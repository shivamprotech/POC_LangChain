import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { createClient, createCluster } from "redis";
import { Document } from "langchain/document";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RedisVectorStore } from "langchain/vectorstores/redis";

export const readPdf = async (filename) => {
  const loader = new PDFLoader(filename);

  return await loader.load();
};

export const uploadDocs = async (docs) => {
  const client = createClient({
    url: process.env.REDIS_URL ?? "redis://localhost:6379",
  });
  await client.connect();

  const vectorStore = await RedisVectorStore.fromDocuments(
    docs,
    new OpenAIEmbeddings({
      openAIApiKey: "OPEN_API_KEY",
    }),
    {
      redisClient: client,
      indexName: "docs",
    }
  );
  await client.disconnect();
};
