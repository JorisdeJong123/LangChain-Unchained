from langchain.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from examples import examples
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from llm_functions import *
import streamlit as st
import os

# Set your OpenAI API Key.
openai_api_key = os.environ.get('OPENAI_API_KEY')

with st.container():
    st.markdown("""
    
# LangChain Unchained - Day 2
## Prompt Generator
In this Streamlit application, we are demonstrating how to build an interactive prompt generator.

We've utilized LangChain, a powerful tool that aids in the generation of applications using language models. LangChain provides a set of components that streamline the process of creating and formatting prompts for language models, and this application showcases a straightforward implementation of these components.

Here's how this interactive prompt generator operates:

- Users enter an initial prompt, which serves as the seed for the language model's creative process.
- The application then uses LangChain to create a more refined and contextualized prompt, drawing from a set of predefined examples.
- These examples are selected based on their semantic similarity to the user's initial prompt, ensuring the output is relevant and focused.
- The final, improved prompt is then displayed on the user interface.

This interactive generator is part of the 'LangChain Unchained' series, where we explore the different facets of using LangChain for language model prompt generation.
                
Check out the explanation of the code on my [Twitter](https://twitter.com/JorisTechTalk)"""
)

with st.container():
    st.markdown("""
                ## Enter initial prompt here:
                """)
initial_prompt = st.text_area(label="Prompt Input", label_visibility='collapsed', placeholder="Generate a workout schedule", key="prompt_input")

if initial_prompt:
    if initial_prompt == "empty":
        st.write("Please enter a valid prompt")
        st.stop()        
    else:
        with st.spinner("Generating Prompt..."):
            # Create an example selector. This is used to select examples that are similar to the input prompt.
            example_selector = create_example_selector(examples, openai_api_key, number_of_examples = 1)

            # Create a prompt template. This is used to format the prompt.
            prompt_template = create_prompt_template(example_selector)

            # Format the prompt with the given initial prompt provided by the user.
            formatted_prompt = format_prompt(prompt_template, initial_prompt)

            # Initialize the language model. This is the model that will generate the output.
            llm = initialize_llm(openai_api_key, model_name = "gpt-3.5-turbo", temperature = 0.2)

            # Initialize the LLMChain.
            llm_chain = initialize_llm_chain(llm, formatted_prompt)

            # Generate the improved prompt.
            improved_prompt = generate_improved_prompt(llm_chain)
            st.write(improved_prompt)