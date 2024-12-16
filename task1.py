import os
import faiss
import openai
import numpy as np
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path, pages=[0]):
    text = ""
    reader = PdfReader(pdf_path)
    for page_num in pages:
        text += reader.pages[page_num].extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def create_openai_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        embeddings.append(response['data'][0]['embedding'])
    return np.array(embeddings, dtype='float32')

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_faiss_index(index, query, vectorizer, chunks):
    query_embedding = openai.Embedding.create(
        input=query,
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]

    query_embedding = np.array(query_embedding, dtype='float32').reshape(1, -1)
    distances, indices = index.search(query_embedding, 3)
    return [chunks[i] for i in indices[0]]

def generate_response_with_llm(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
    You are a helpful assistant. Answer the following question based on the provided context:
    
    Context:
    {context}
    
    Question:
    {query}
    
    Provide an accurate and detailed answer based on the data provided.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def extract_table_from_page(pdf_path, page_num=0):
    reader = PdfReader(pdf_path)
    page = reader.pages[page_num]
    text = page.extract_text()
    lines = text.split("\n")
    table_data = [line.split() for line in lines if line.strip()]
    df = pd.DataFrame(table_data)
    return df

def compare_data_across_pdfs(pdf_paths, pages, comparison_query):
    combined_text = ""
    for pdf_path in pdf_paths:
        combined_text += extract_text_from_pdf(pdf_path, pages)
    chunks = chunk_text(combined_text)
    embeddings = create_openai_embeddings(chunks)
    index = build_faiss_index(embeddings)
    
    retrieved_chunks = search_faiss_index(index, comparison_query, None, chunks)
    response = generate_response_with_llm(comparison_query, retrieved_chunks)
    return response

if __name__ == "__main__":

    openai.api_key = "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"  # Replace with your OpenAI API Key

    pdf_paths = [
        "C:/Users/onepr/Downloads/Tables- Charts- and Graphs with Examples from History- Economics- Education- Psychology- Urban Affairs and Everyday Life - 2017-2018.pdf",  # Replace with actual PDF paths
        "C:/Users/onepr/Downloads/Tables- Charts- and Graphs with Examples from History- Economics- Education- Psychology- Urban Affairs and Everyday Life - 2017-2018.pdf" 
    ]

    for path in pdf_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"PDF file not found: {path}")

    print("Extracting text and building FAISS index...")
    all_chunks = []
    all_embeddings = []

    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path}...")
        text = extract_text_from_pdf(pdf_path, pages=[1, 5])  # Pages to extract
        chunks = chunk_text(text)
        embeddings = create_openai_embeddings(chunks)
        all_chunks.extend(chunks)
        all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings)
    index = build_faiss_index(all_embeddings)

    user_query = input("Please enter your query: ") 
    print("Searching for relevant chunks...")
    retrieved_chunks = search_faiss_index(index, user_query, None, all_chunks)

    print("Generating response...")
    final_response = generate_response_with_llm(user_query, retrieved_chunks)
    print("Response:")
    print(final_response)

    comparison_query = input("Please enter your comparison query: ")  # Taking user input for the comparison query
    print("\nPerforming comparison query...")
    comparison_response = compare_data_across_pdfs(pdf_paths, pages=[1, 5], comparison_query=comparison_query)
    print("Comparison Response:")
    print(comparison_response)

    print("\nExtracting tabular data from first PDF (page 6)...")
    table_data = extract_table_from_page(pdf_paths[0], page_num=5)
    print("Tabular Data:")
    print(table_data)
