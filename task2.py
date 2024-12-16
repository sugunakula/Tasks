import os
import openai
import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from typing import List, Tuple
from tqdm import tqdm
import tiktoken

OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

DIMENSIONS = 1536
index = faiss.IndexFlatL2(DIMENSIONS)

def fetch_web_content(url: str) -> str:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract text from paragraphs
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return ""

def split_text(text: str, max_tokens: int = 500) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    response = openai.Embedding.create(input=text, model=model)
    return response['data'][0]['embedding']

def ingest_website(urls: List[str]) -> Tuple[List[str], List[np.ndarray]]:
    all_chunks = []
    all_embeddings = []

    print("Ingesting websites...")
    for url in urls:
        print(f"Fetching content from {url}...")
        content = fetch_web_content(url)

        if content:
            print("Splitting content into chunks...")
            chunks = split_text(content)

            print("Generating embeddings...")
            for chunk in tqdm(chunks):
                embedding = get_embedding(chunk)
                all_chunks.append(chunk)
                all_embeddings.append(np.array(embedding, dtype=np.float32))

    if all_embeddings:
        embeddings_matrix = np.vstack(all_embeddings)
        index.add(embeddings_matrix)

    return all_chunks, all_embeddings

def search_similar_chunks(query: str, chunks: List[str], top_k: int = 5) -> List[str]:
    print("Generating query embedding...")
    query_embedding = np.array(get_embedding(query), dtype=np.float32).reshape(1, -1)

    print("Searching for similar chunks...")
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]

    return results

def generate_response(query: str, context: List[str]) -> str:
    prompt = f"""
    Context:
    {"\n".join(context)}

    Question: {query}

    Based on the above context, provide a detailed and accurate answer.
    """
    print("Generating response from GPT-4...")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful and factual assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

def main():
    websites = [
        "https://www.uchicago.edu/",
        "https://www.washington.edu/",
        "https://www.stanford.edu/",
        "https://und.edu/"
    ]

    chunks, _ = ingest_website(websites)

    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        similar_chunks = search_similar_chunks(query, chunks)

        response = generate_response(query, similar_chunks)

        print("\nResponse:")
        print(response)

if __name__ == "__main__":
    main()
