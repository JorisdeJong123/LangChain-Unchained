from langchain.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from examples import examples
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain import LLMChain
import os
from langchain.embeddings import OpenAIEmbeddings

# Function to create an example prompt selector.
def create_example_selector(examples, openai_api_key, number_of_examples):
    example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(openai_api_key=openai_api_key),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=number_of_examples
    )
    
    return example_selector  


# Function to create a prompt template.
def create_prompt_template(example_selector):
    example_prompt = PromptTemplate(input_variables=["question", "answer"], template="Question: {question}\n{answer}")
    prompt = FewShotPromptTemplate(
        example_selector=example_selector, 
        example_prompt=example_prompt, 
        prefix="""
    You are an expert in writing prompts for large language models. 

    You're goal is to rewrite prompts for gaining better results.

    Here are several tips on writing great prompts:

    -------

    Start the prompt by stating that it is an expert in the subject.

    Put instructions at the beginning of the prompt and use ### or to separate the instruction and context 

    Be specific, descriptive and as detailed as possible about the desired context, outcome, length, format, style, etc 

    Articulate the desired output format through examples (example 1, example 2). 

    Reduce “fluffy” and imprecise descriptions

    Instead of just saying what not to do, say what to do instead.

    -------

    Here's an example of what the input question will look like with the corresponding result

        """,
        suffix="This is the prompt you need to reform: Question: {input} \nAnswer:" ,
        input_variables=["input"]
    )
    return prompt

# Function to format the prompt.
def format_prompt(prompt, question):
    formatted_prompt = prompt.format(input=question)
    return formatted_prompt

# Function to initialize the large language model.
def initialize_llm(openai_api_key, model_name, temperature):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=temperature)
    return llm

# Function to initialize the LLMChain.
def initialize_llm_chain(llm, prompt):
    llm_chain = LLMChain(
        llm=llm, 
        prompt=PromptTemplate.from_template(prompt)
    )
    return llm_chain

# Function to generate improved prompt
def generate_improved_prompt(llm_chain):
    answer = llm_chain.predict()
    return answer