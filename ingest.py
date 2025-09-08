import os
from langchain.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader, TextLoader, CSVLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path

# --- Configuration: The 'settings' for our data ingestion process ---
# We define our file paths and database name here so they're easy to change.
DOCS_FOLDER = Path("docs")
VECTOR_DB_PATH = "chroma_db"

# This dictionary acts as a smart lookup table. It maps each file extension
# to the correct LangChain document loader class and any specific arguments it needs.
# It ensures we use the right tool for each file type.
FILE_LOADERS = {
    ".pdf": (PyPDFLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".txt": (TextLoader, {'encoding': 'utf-8'}),
    ".csv": (CSVLoader, {'encoding': 'utf-8'}),
    # This JSON loader is configured to extract a specific schema (questions and answers).
    # This is a key design choice for our specific dataset.
    ".json": (JSONLoader, {'jq_schema': ".[].question, .[].answer", "text_content": False}),
}

def ingest_documents():
    """
    This is the main function for our data pipeline. It orchestrates the entire process
    of loading, chunking, and indexing our documents. It's the "ETL" (Extract, Transform, Load)
    for our RAG system's knowledge base.
    """
    if not DOCS_FOLDER.exists() or not DOCS_FOLDER.is_dir():
        # A simple sanity check to make sure the 'docs' folder is where we expect it to be.
        print(f"Error: The '{DOCS_FOLDER}' directory does not exist or is not a directory.")
        return

    documents = []
    
    # We loop through every file in the specified 'docs' folder.
    for filepath in DOCS_FOLDER.iterdir():
        if filepath.is_file():
            file_extension = filepath.suffix.lower()
            
            # Check if we have a supported loader for this file type.
            if file_extension in FILE_LOADERS:
                loader_class, loader_args = FILE_LOADERS[file_extension]
                loader = loader_class(str(filepath), **loader_args)
                
                try:
                    loaded_docs = loader.load()
                    print(f"Successfully loaded {len(loaded_docs)} pages from {filepath.name}.")
                    
                    # This is a critical step for our citation system!
                    # While PyPDFLoader naturally extracts page numbers,
                    # other loaders (like for JSON and TXT) do not.
                    # We manually add a 'page' number to the metadata for non-PDFs
                    # so that citations work consistently across all document types.
                    if file_extension != '.pdf':
                        for i, doc in enumerate(loaded_docs):
                            # We use a simple index as the 'page' number.
                            doc.metadata['page'] = i  
                            
                    documents.extend(loaded_docs)
                except Exception as e:
                    # We print a helpful error message but continue processing other files.
                    print(f"Error loading {filepath.name}: {e}")
            else:
                # We simply skip any file types we don't know how to handle.
                print(f"Skipping unsupported file: {filepath.name}")

    if not documents:
        print("No supported documents found to ingest.")
        return

    # --- Chunking Strategy: Breaking down big documents into digestible pieces ---
    print(f"Total documents loaded: {len(documents)}")
    # The RecursiveCharacterTextSplitter is a smart chunker. It tries to split text
    # on different characters (like newlines, sentences) to keep related text together.
    # We've chosen a chunk size of 1000 characters and an overlap of 200 to ensure
    # each chunk has enough context but isn't too large for the LLM's context window.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from the documents.")
    
    # --- Embedding and Storing: Converting text to numbers for the vector database ---
    # We use a powerful but lightweight Sentence Transformer model to convert our
    # text chunks into dense numerical vectors (embeddings). This model runs locally
    # and doesn't require an API key.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("Creating and persisting vector store...")
    # This is the final step. We take our list of chunks and their corresponding embeddings
    # and store them in a persistent Chroma vector database on our local machine.
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH,
    )
    print(f"Vector store created successfully at '{VECTOR_DB_PATH}'.")

if __name__ == "__main__":
    # The entry point for the script. When you run `python ingest.py`, this is what gets executed.
    ingest_documents()