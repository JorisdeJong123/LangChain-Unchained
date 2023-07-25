import openai
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from audio_functions import NamedBytesIO, transcribe_audio
from prompts import PROMPT_SUMMARY, REFINE_PROMPT_SUMMARY
from llm_functions import initialize_llm, initialize_summarize_chain, split_text

# Set your OpenAI API Key.
openai_api_key = 'YOUR_API_KEY'

# Set your OpenAI API Key for Whisper.
openai.api_key = 'YOUR_API_KEY'

with st.container():
    st.markdown("""

With this new app, 'Whisper Notes,' you can effortlessly transform your recorded meetings or general thoughts into comprehensive meeting notes. This Streamlit application leverages the power of OpenAI Whisper for accurate audio transcription and employs LangChain's capabilities to post-process the transcriptions into detailed meeting summaries.

Here's how 'Whisper Notes' works:

1. **Record and Transcribe:** Using your phone or laptop, you can record meetings or any spoken content you want to capture. OpenAI Whisper, a cutting-edge automatic speech recognition system, processes the audio and converts it into text with remarkable accuracy.

2. **Generate Meeting Summaries:** The transcribed text is then passed through LangChain, a robust tool specialized in language model prompt generation. LangChain refines and structures the text, converting it into detailed and well-organized meeting summaries.

3. **Efficient and Effective:** 'Whisper Notes' streamlines the process of note-taking, saving you valuable time and effort. Instead of manually transcribing and summarizing lengthy recordings, you can rely on the power of AI to produce comprehensive meeting notes automatically.

4. **Versatile Use:** Whether you're a professional in a corporate setting or a student attending lectures, 'Whisper Notes' is a versatile tool that caters to various scenarios. Record your ideas, brainstorming sessions, or important meetings, and transform them into actionable and easy-to-digest notes.

5. **Seamless User Experience:** The user interface of 'Whisper Notes' is designed to be intuitive and user-friendly. With just a few clicks, you can turn your audio recordings into detailed meeting summaries effortlessly.

6. **Privacy and Security:** We understand the importance of data privacy. Rest assured, your recordings and transcriptions are processed securely and won't be shared with any third parties.

'Whisper Notes' is part of the ongoing exploration of the capabilities of OpenAI Whisper and LangChain. Experience the convenience and productivity of automatic meeting note generation today!

Check out the explanation of the code and the development process on my [Twitter](https://twitter.com/JorisTechTalk).
                """)

with st.container():
    # Experimental Streamlit feature to record audio. Set the pause_threshold to the amount of silent seconds you want to wait before the recording stops.
    audio_bytes = audio_recorder(text= "Record your meeting" ,pause_threshold=5)
    if audio_bytes:
        st.audio(audio_bytes)

        # Convert the audio bytes to a NamedBytesIO object.
        audio_bytes = NamedBytesIO(audio_bytes, name="audio.mp3")

        # Get response from OpenAI Whisper API. 
        response = transcribe_audio(audio_bytes=audio_bytes, openai_api_key=openai_api_key)

        # Assign the transcript to a variable
        transcript = response["text"]

        # Split the transcript into chunks.
        transcript_chunks = split_text(data=transcript, chunk_size=2000, chunk_overlap=50)

        # Initialize the large language model.
        llm = initialize_llm(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k", temperature=0.2)

        # Initialize the summarize chain.
        summarize_chain = initialize_summarize_chain(llm=llm, chain_type="refine", question_prompt=PROMPT_SUMMARY, refine_prompt=REFINE_PROMPT_SUMMARY)

        # Run the summarize chain.
        summary = summarize_chain.run(transcript_chunks)

        # Print the summary. Use st.code to display the summary in a code block for easy copy/paste.
        st.code(summary)



