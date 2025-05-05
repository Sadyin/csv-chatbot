import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import openai
import os

class ChromaHelper:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        self.collection = self.client.get_or_create_collection(
            name="csv_data",
            embedding_function=self.openai_ef
        )

    def ingest_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
            documents, metadatas, ids = [], [], []

            for idx, row in df.iterrows():
                text = " | ".join([f"{k}:{str(v)}" for k, v in row.items()])
                documents.append(text)
                metadatas.append({"row_id": idx, "source": csv_path})
                ids.append(f"{os.path.basename(csv_path)}_{idx}")

                if len(documents) % 100 == 0:
                    self.collection.add(
                        documents=documents,
                        metadatas=metadatas,
                        ids=ids
                    )
                    documents, metadatas, ids = [], [], []

            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )

            return True, f"Processed {len(df)} rows"
        except Exception as e:
            return False, str(e)

    def search(self, query, n_results=5):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return "\n".join([f"• {doc}" for doc in results['documents'][0]])

    def call_openai(self, prompt):
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
