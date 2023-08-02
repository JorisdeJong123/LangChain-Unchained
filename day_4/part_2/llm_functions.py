from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from prompts import PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader 

# Function to load data from PDF
def load_data(uploaded_file):
    # Load data from PDF
    pdf_reader = PdfReader(uploaded_file)
    # Combine text from Document into one string
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    return text

# Function to split text into chunks
def split_text(text, chunk_size, chunk_overlap):
    # Initialize text splitter
    text_splitter = TokenTextSplitter(model_name="gpt-3.5-turbo-16k", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Split text into chunks
    text_chunks = text_splitter.split_text(text)

    # Convert chunks to documents
    documents = [Document(page_content=t) for t in text_chunks]

    return documents

# Function to initialize large language model
def initialize_llm(openai_api_key, model, temperature):
    # Initialize large language model
    llm = ChatOpenAI(openai_api_key=openai_api_key,model=model, temperature=temperature)

    return llm

# Function to generate questions
def generate_questions(llm, chain_type, documents):
    # Initialize question chain
    question_chain = load_summarize_chain(llm=llm, chain_type=chain_type, question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)

    # Generate questions
    questions = question_chain.run(documents)

    return questions

# Function to create Retrieval QA Chain
def create_retrieval_qa_chain(openai_api_key, documents, llm):
    # Set embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create vector database
    vector_database = Chroma.from_documents(documents=documents, embedding=embeddings)

    # Create Retrieval QA Chain
    retrieval_qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_database.as_retriever())

    return retrieval_qa_chain