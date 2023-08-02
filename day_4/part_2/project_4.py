import streamlit as st
from llm_functions import load_data, split_text, initialize_llm, generate_questions, create_retrieval_qa_chain

# Initialization of session states
# Since Streamlit always reruns the script when a widget changes, we need to initialize the session states
if 'questions' not in st.session_state:
    st.session_state['questions'] = 'empty'
    st.session_state['seperated_question_list'] = 'empty'
    st.session_state['questions_to_answers'] = 'empty'
    st.session_state['submitted'] = 'empty'

with st.container():
    st.markdown("""# Project 4: Smart Study Buddy""")

# Get user's OpenAI API Key
openai_api_key = st.text_input(label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input", help="How to get an OpenAI API Key: https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/")

# Container for markdown text
with st.container():
    st.markdown("""Make sure you've entered your OpenAI API Key. 
                Don't have an API key yet? 
                Read [this](https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/) article on how to get an API key.""")
    
# Let user upload a file
uploaded_file = st.file_uploader("Upload your study material", type=['pdf'])

if uploaded_file is not None:
    # Check whether user entered an API key
    if not openai_api_key:
        st.error("Please enter your OpenAI API Key")
    else:
        # Load data from PDF
        text_from_pdf = load_data(uploaded_file)

        # Split text for question generation
        documents_for_question_gen = split_text(text_from_pdf, chunk_size=10000, chunk_overlap=200)

        # Split text for question answering
        documents_for_question_answering = split_text(text_from_pdf, chunk_size=500, chunk_overlap=200)

        # Initialize large language model for question generation
        llm_question_gen = initialize_llm(openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k", temperature=0.4)

        # Initialize large language model for question answering
        llm_question_answering = initialize_llm(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0.1)

        # Create questions if they have not yet been generated
        if st.session_state['questions'] == 'empty':
            with st.spinner("Generating questions..."):
                # Assign the generated questions to the session state. This way, the questions are only generated once.
                st.session_state['questions'] = generate_questions(llm=llm_question_gen, chain_type="refine", documents=documents_for_question_gen)

        if st.session_state['questions'] != 'empty':
            # Show questions on screen. You could use st.code for easy copy-pasting.
            st.info(st.session_state['questions'])

            # Split questions into a list
            st.session_state['questions_list'] = st.session_state['questions'].split('\n')

            with st.form(key='my_form'):
                # Create a list of questions that have to be answered
                st.session_state['questions_to_answers'] = st.multiselect(label="Select questions to answer", options=st.session_state['questions_list'])
                submitted = st.form_submit_button('Generate answers')
                if submitted:
                    st.session_state['submitted'] = True

            if st.session_state['submitted']:
            # Initialize the Retrieval QA Chain
                with st.spinner("Generating answers..."):
                    generate_answer_chain = create_retrieval_qa_chain(openai_api_key=openai_api_key, documents=documents_for_question_answering, llm=llm_question_answering)
                    # For each question, generate an answer
                    for question in st.session_state['questions_to_answers']:
                        # Generate answer
                        answer = generate_answer_chain.run(question)
                        # Show answer on screen
                        st.write(f"Question: {question}")
                        st.info(f"Answer: {answer}")