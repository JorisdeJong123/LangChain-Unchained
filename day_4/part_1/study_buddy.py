from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from part_2.prompts import PROMPT_QUESTIONS, REFINE_PROMPT_QUESTIONS
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Set OpenAI API key
openai_api_key = 'YOUR_OPENAI_API_KEY'

# Set file path
file_path = 'path/to/file.pdf'

# Load data from PDF
loader = PyPDFLoader(file_path)
data = loader.load()

# Combine text from Document into one string for question generation
text_question_gen = ''
for page in data:
    text_question_gen += page.page_content

# Initialize Text Splitter for question generation
text_splitter_question_gen = TokenTextSplitter(model_name="gpt-3.5-turbo-16k", chunk_size=10000, chunk_overlap=200)

# Split text into chunks for question generation
text_chunks_question_gen = text_splitter_question_gen.split_text(text_question_gen)

# Convert chunks into Documents for question generation
docs_question_gen = [Document(page_content=t) for t in text_chunks_question_gen]

# Initialize Text Splitter for answer generation
text_splitter_answer_gen = TokenTextSplitter(model_name="gpt-3.5-turbo-16k", chunk_size=1000, chunk_overlap=100)

# Split documents into chunks for answer generation
docs_answer_gen = text_splitter_answer_gen.split_documents(docs_question_gen)

# Initialize Large Language Model for question generation
llm_question_gen = ChatOpenAI(openai_api_key=openai_api_key,temperature=0.4, model="gpt-3.5-turbo-16k")

# Initialize question generation chain
question_gen_chain = load_summarize_chain(llm = llm_question_gen, chain_type = "refine", verbose = True, question_prompt=PROMPT_QUESTIONS, refine_prompt=REFINE_PROMPT_QUESTIONS)

# Run question generation chain
questions = question_gen_chain.run(docs_question_gen)

# Initialize Large Language Model for answer generation
llm_answer_gen = ChatOpenAI(openai_api_key=openai_api_key,temperature=0.1, model="gpt-3.5-turbo-16k")

# Create vector database for answer generation
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize vector store for answer generation
vector_store = Chroma.from_documents(docs_answer_gen, embeddings)

# Initialize retrieval chain for answer generation
answer_gen_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff", retriever=vector_store.as_retriever(k=2))

# Split generated questions into a list of questions
question_list = questions.split("\n")

# Answer each question and save to a file
for question in question_list:
    print("Question: ", question)
    answer = answer_gen_chain.run(question)
    print("Answer: ", answer)
    print("--------------------------------------------------\n\n")
    # Save answer to file
    with open("answers.txt", "a") as f:
        f.write("Question: " + question + "\n")
        f.write("Answer: " + answer + "\n")
        f.write("--------------------------------------------------\n\n")

