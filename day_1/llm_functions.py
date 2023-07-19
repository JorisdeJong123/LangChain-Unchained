from langchain import LLMChain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

# Function to load a YouTube video and get the transcript.
def load_youtube_video(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    data = loader.load()
    return data

# Function to split the transcript into chunks.
def split_text(data, chunk_size, chunk_overlap):
    text_splitter = TokenTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    docs = text_splitter.split_documents(data)
    return docs

# Function to initialize the large language model.
def initialize_llm(openai_api_key, model_name, temperature):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=temperature)
    return llm

# Function to initialize the summarize chain.
def initialize_summarize_chain(llm, chain_type, question_prompt, refine_prompt):
    strategy_chain = load_summarize_chain(llm=llm, chain_type=chain_type, verbose=True, question_prompt=question_prompt, refine_prompt=refine_prompt)
    return strategy_chain

# Function to generate a strategy.
def generate_strategy(strategy_chain, docs):
    strategy = strategy_chain.run(docs)
    return strategy

# Function to initialize the plan chain.
def initialize_plan_chain(llm, prompt, verbose):
    plan_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)
    return plan_chain

# Function to generate a plan.
def generate_plan(plan_chain, strategy):
    plan = plan_chain(strategy)
    return plan