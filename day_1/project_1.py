import streamlit as st
from prompts import PROMPT_STRATEGY, PROMPT_STRATEGY_REFINE, PROMPT_PLAN
from llm_functions import load_youtube_video, split_text, initialize_llm, initialize_summarize_chain, generate_strategy, initialize_plan_chain, generate_plan
import os

# Set your OpenAI API Key.
openai_api_key = os.environ.get('OPENAI_API_KEY')

with st.container():
    st.markdown("""
    
# LangChain Unchained - Day 1
This Streamlit Implementation shows how to create a strategy for a four-hour workday based on a YouTube video.

We're using an easy LangChain implementation to show how to use the different components of LangChain.
                
This is part of my 'LangChain Unchained' series. Check out the explanation about the code on my [Twitter](https://twitter.com/JorisTechTalk)"""
)
                
url_input = st.text_input(label="URL Input", label_visibility='collapsed', placeholder="https://www.youtube.com/watch?v=T6hmdrsLQj8", key="url_input")

if url_input:
    if url_input == 'empty':
        st.write("Please enter a valid URL")
        st.stop()
    else:
        with st.spinner('Loading the YouTube video transcript...'):
            # Load the YouTube video transcript.
            data = load_youtube_video(url_input)

            # Split the transcript into chunks.
            docs = split_text(data, chunk_size=1000, chunk_overlap=100)

        # Initialize the large language model.
        llm = initialize_llm(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0.4)

        with st.spinner('Generating a strategy...'):
            # Initialize the summarize chain.
            strategy_chain = initialize_summarize_chain(llm=llm, chain_type='refine', question_prompt=PROMPT_STRATEGY, refine_prompt=PROMPT_STRATEGY_REFINE)

            # Generate a strategy.
            strategy = generate_strategy(strategy_chain, docs)

        with st.spinner('Generating a plan...'):
            # Initialize the plan chain.
            plan_chain = initialize_plan_chain(llm=llm, prompt=PROMPT_PLAN, verbose=True)

            # Generate a plan based on the strategy.
            plan = generate_plan(plan_chain, strategy)
            st.write(plan)
        st.stop()
