from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document

# Function to initialize the large language model.
def initialize_llm(openai_api_key, model_name, temperature):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=temperature)
    return llm

# Function to initialize the summarize chain.
def initialize_summarize_chain(llm, chain_type, question_prompt, refine_prompt):
    strategy_chain = load_summarize_chain(llm=llm, chain_type=chain_type, verbose=True, question_prompt=question_prompt, refine_prompt=refine_prompt)
    return strategy_chain

# Function to split the transcript into chunks.
def split_text(data, chunk_size, chunk_overlap):
    text_splitter = TokenTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    texts = text_splitter.split_text(data)

    # Create documents for further processing
    docs = [Document(page_content=t) for t in texts]
    return docs

