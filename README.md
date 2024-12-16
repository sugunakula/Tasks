# README for Tasks

## Task 1: Chat with PDF Using RAG Pipeline

### Overview
This task focuses on implementing a Retrieval-Augmented Generation (RAG) pipeline to enable interactive queries on semi-structured data in multiple PDF files. The system extracts, chunks, embeds, and stores the data for efficient retrieval, allowing accurate responses to user queries, including comparative analyses.

### Functional Requirements
1. **Data Ingestion**
    - **Input**: PDF files containing semi-structured data.
    - **Process**:
        - Extract text and structured information from PDFs.
        - Segment data into logical chunks for granularity.
        - Generate vector embeddings using a pre-trained embedding model.
        - Store embeddings in a vector database for similarity-based retrieval.

2. **Query Handling**
    - **Input**: User’s natural language query.
    - **Process**:
        - Convert the query into vector embeddings using the same embedding model.
        - Perform a similarity search in the vector database to retrieve relevant chunks.
        - Pass retrieved chunks to the LLM (Large Language Model) along with contextual prompts to generate detailed responses.

3. **Comparison Queries**
    - **Input**: Queries requiring a comparison.
    - **Process**:
        - Extract terms or fields for comparison across multiple PDFs.
        - Retrieve relevant chunks from the vector database.
        - Aggregate data and format responses (e.g., tabular or bullet points).

4. **Response Generation**
    - **Input**: Retrieved information and user query.
    - **Process**:
        - Use the LLM with retrieval-augmented prompts to generate responses.
        - Ensure factuality by directly incorporating retrieved data.

### Example Data
1. **Unemployment Information**: Retrieve exact data based on the degree type from page 2 of the sample PDF.
2. **Tabular Data**: Extract and present tabular data from page 6.

### Resources
Sample PDF: [Tables, Charts, and Graphs Examples](https://www.hunter.cuny.edu/dolciani/pdf_files/workshop-materials/mmc-presentations/tables-charts-and-graphs-with-examples-from.pdf)

---

## Task 2: Chat with Website Using RAG Pipeline

### Overview
This task involves creating a Retrieval-Augmented Generation (RAG) pipeline to interact with structured and unstructured data from websites. The system crawls, scrapes, processes website content, and stores it as embeddings for efficient query handling and response generation.

### Functional Requirements
1. **Data Ingestion**
    - **Input**: URLs or lists of websites to crawl/scrape.
    - **Process**:
        - Crawl and scrape content from specified websites.
        - Extract key data fields, metadata, and textual content.
        - Segment content into chunks for better granularity.
        - Generate vector embeddings using a pre-trained embedding model.
        - Store embeddings in a vector database with associated metadata.

2. **Query Handling**
    - **Input**: User’s natural language query.
    - **Process**:
        - Convert the query into vector embeddings using the same embedding model.
        - Perform a similarity search in the vector database to retrieve relevant chunks.
        - Pass retrieved chunks to the LLM along with contextual prompts for response generation.

3. **Response Generation**
    - **Input**: Retrieved information and user query.
    - **Process**:
        - Use the LLM with retrieval-augmented prompts to produce responses.
        - Ensure responses are context-rich and factually accurate by directly integrating retrieved data.

### Example Websites
1. [University of Chicago](https://www.uchicago.edu/)
2. [University of Washington](https://www.washington.edu/)
3. [Stanford University](https://www.stanford.edu/)
4. [University of North Dakota](https://und.edu/)

---

## Common Technologies and Tools
- **Embedding Model**: Pre-trained models for vector representation of text.
- **Vector Database**: For efficient similarity-based retrieval.
- **LLM**: Used for response generation and conversational interactions.

## Installation and Setup
1. Clone the repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure the vector database and LLM API keys in the `.env` file.

## Usage
- For Task 1:
  - Upload PDF files to the data ingestion module.
  - Query the system with natural language questions.
- For Task 2:
  - Provide website URLs to the data ingestion module.
  - Query the system with context-specific questions.

## License
This project is licensed under the MIT License.

