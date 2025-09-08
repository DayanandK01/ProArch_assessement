import streamlit as st
import os
import json
import time
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer

# --- Configuration: The app's global settings and secrets ---
# We get the API key from Streamlit's secrets management, which is a best practice for security.
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
VECTOR_DB_PATH = "chroma_db"
LOG_FILE = "logs.json"

# --- Token Counting Function: Estimating our model's cost and usage ---
# We use Streamlit's cache_resource decorator so the tokenizer is only loaded once
# when the app starts, which saves a lot of time and memory.
@st.cache_resource
def get_tokenizer():
    """Loads a generic tokenizer to help us estimate the number of tokens in our text."""
    # A generic tokenizer like this is useful for approximating costs and performance.
    return AutoTokenizer.from_pretrained("google/flan-t5-large")

tokenizer = get_tokenizer()

def count_tokens(text):
    """A simple helper function to count tokens using the tokenizer we loaded."""
    return len(tokenizer.encode(text))

# --- Core Components: The LLM and Vector Database setup ---
@st.cache_resource
def load_components():
    """
    Loads our core AI components: the LLM and the vector store.
    This function is cached to prevent these expensive components from reloading on every interaction.
    """
    # We configure our LLM, in this case, Google's Gemini-1.5-Flash model.
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.1,  # A low temperature makes the model's output more factual and less creative.
        google_api_key=GOOGLE_API_KEY,
        max_output_tokens=2048, # We give it a generous token limit to allow for detailed answers.
        # We relax the safety settings. This is a common practice in RAG apps, as the
        # retrieved documents might contain content that would otherwise be flagged.
        safety_settings={
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    # We load the same local embedding model used in the 'ingest.py' script.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # This connects to our local Chroma vector store, which holds the embeddings of our documents.
    vector_store = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    return llm, vector_store

# We call the function to load the components at the start of the script.
llm, vector_store = load_components()

# --- RAG Chain and Prompt: The brain of our application ---
@st.cache_resource
def get_qa_chain():
    """
    Creates the LangChain RetrievalQA chain. This chain is what ties everything together:
    it takes the user's question, retrieves relevant chunks from the vector store,
    and passes them to the LLM with our custom prompt to generate an answer.
    """
    # This is our custom prompt template. It's the "personality" and "instructions" for our AI.
    # It explicitly tells the LLM to only use the provided context, cite sources, and
    # respond with "I don't know" if the information isn't available.
    prompt_template = """
    You are an AI assistant for a set of internal documents. Your task is to provide comprehensive and accurate answers based *only* on the context provided.
    
    If the information needed to answer the question is not present in the context, state "I don't know" or "The information is not in the provided documents." Do not invent an answer.
    
    When answering, cite the sources by mentioning the document name and page number. Format citations as a list at the end of the answer. Example: Source: `document_name.pdf` (Page: X)
    
    **Instructions for Response:**
    * **Elaborate and provide a detailed, lengthy explanation of the concept.**
    * **Structure your answer in a clear and well-organized paragraph form.**
    * **Do not use bullet points or lists in your main answer. Save them for citations.**

    Context:
    {context}

    Question: {question}
    """
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # 'Stuff' means we 'stuff' all the retrieved documents into one large prompt.
        retriever=vector_store.as_retriever(),
        return_source_documents=True, # This is essential! It gives us access to the metadata for citations.
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

qa_chain = get_qa_chain()

# --- Logging: Recording every query for analysis and debugging ---
def log_query(query, response, sources, query_tokens, response_tokens, response_time):
    """
    This function appends a detailed log of each user query and the model's response
    to a JSON file. This is crucial for tracking performance and user interactions.
    """
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "response": response,
        "query_tokens": query_tokens,
        "response_tokens": response_tokens,
        "retrieved_doc_ids": [doc.metadata.get('source', 'N/A') for doc in sources],
        "retrieved_passages": [doc.page_content for doc in sources],
        "response_time_sec": response_time
    }
    
    # We first try to load existing logs. If the file is not found or is corrupted, we start a new list.
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []
        
    logs.append(log_entry)
    
    # We save the updated list of logs back to the JSON file with a readable format.
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)

# --- Streamlit UI: The interactive front-end of our application ---
st.title("GetInfo - AI-Powered Document Q&A")
st.caption("Answers grounded in a specific document set.")

# This "health check" button is a simple way to test the app's functionality with predefined queries.
if st.button("Run Test Queries (Health Check)"):
    st.write("Running sample queries...")
    sample_queries = [
        "What are the main features of product X?",
        "Is there any information about the 2025 financial projections?",
        "What is the capital of France?"  # A key test: an out-of-scope question to check for hallucination.
    ]
    
    for q in sample_queries:
        st.info(f"Test Query: {q}")
        with st.spinner("Thinking..."):
            start_time = time.time()
            result = qa_chain({"query": q})
            end_time = time.time()
            
            response_text = result["result"]
            sources = result["source_documents"]
            
            query_tokens = count_tokens(q)
            response_tokens = count_tokens(response_text)
            
            st.write(response_text)
            log_query(q, response_text, sources, query_tokens, response_tokens, end_time - start_time)
            
            st.subheader("Citations")
            for doc in sources:
                st.write(f"- Source: `{doc.metadata.get('source', 'N/A')}` (Page: {doc.metadata.get('page', 'N/A')})")

st.header("Ask a Question")

# Define the list of sample questions
sample_questions = [
    "What are the core architectural components and training methods behind Large Language Models (LLMs)?",
    "How do agentic AI systems differ from traditional AI models in terms of autonomy and decision-making?",
    "What are the main challenges in fine-tuning LLMs, and how does Reinforcement Learning from Human Feedback (RLHF) improve model alignment?",
    "Explain how advances like retrieval-augmented generation (RAG) and mixture-of-experts models improve LLM capabilities and efficiency.",
    "Describe key safety and ethical considerations when deploying autonomous agentic AI applications."
]

# Create a container for the sample questions
with st.sidebar:
    st.header("Questions To Ask")
    for q in sample_questions:
        if st.button(q):
            st.session_state.user_query = q

# Check if a question was clicked and use it to pre-fill the text input
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
    
user_query = st.text_input("Enter your question here:", value=st.session_state.user_query, key="main_query_input")

# This block runs when the user enters a query and hits enter.
if user_query:
    # First, we check if the vector database exists. If not, we instruct the user to run the ingestion script.
    if not os.path.exists(VECTOR_DB_PATH):
        st.error("Vector database not found. Please run the `ingest.py` script first.")
    else:
        with st.spinner("Thinking..."):
            start_time = time.time()
            result = qa_chain({"query": user_query})
            end_time = time.time()
            
            response_text = result["result"]
            sources = result["source_documents"]
            
            query_tokens = count_tokens(user_query)
            response_tokens = count_tokens(response_text)
            
            # Display the final answer to the user.
            st.subheader("Answer")
            st.markdown(response_text)
            
            # Display the citations. This provides transparency and allows users to verify the information.
            st.subheader("Citations")
            for doc in sources:
                st.write(f"- Source: `{doc.metadata.get('source', 'N/A')}` (Page: {doc.metadata.get('page', 'N/A')})")
            
            # This checkbox lets the user inspect the raw chunks that were retrieved from the database.
            # It's a great debugging tool for developers and a way to build user trust.
            if st.checkbox("Show retrieved chunks"):
                st.subheader("Retrieved Chunks")
                for i, doc in enumerate(sources):
                    source_name = doc.metadata.get('source', 'N/A')
                    page_number = doc.metadata.get('page', 'N/A')
                    st.text_area(label=f"Chunk {i+1} from `{source_name}` (Page: {page_number})", value=doc.page_content, height=200, disabled=True)
            
            # Finally, we log the details of this query for our records.
            log_query(user_query, response_text, sources, query_tokens, response_tokens, end_time - start_time)