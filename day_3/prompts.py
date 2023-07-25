from langchain.prompts import PromptTemplate

prompt_template_summary = """
You are a management assistant with a specialization in note taking. You are taking notes for a meeting.

Write a detailed summary of the following transcript of a meeting:


{text}

Make sure you don't lose any important information. Be as detailed as possible in your summary. 

Also end with a list of:

- Main takeaways
- Action items
- Decisions
- Open questions
- Next steps

If there are any follow-up meetings, make sure to include them in the summary and mentioned it specifically.


DETAILED SUMMARY IN ENGLISH:"""

PROMPT_SUMMARY = PromptTemplate(template=prompt_template_summary, input_variables=["text"])

refine_template_summary = (
'''
You are a management assistant with a specialization in note taking. You are taking notes for a meeting.
Your job is to provide detailed summary of the following transcript of a meeting:
We have provided an existing summary up to a certain point: {existing_answer}.
We have the opportunity to refine the existing summary (only if needed) with some more context below.
----------------
{text}
----------------
Given the new context, refine the original summary in English.
If the context isn't useful, return the original summary. Make sure you are detailed in your summary.
Make sure you don't lose any important information. Be as detailed as possible. 

Also end with a list of:

- Main takeaways
- Action items
- Decisions
- Open questions
- Next steps

If there are any follow-up meetings, make sure to include them in the summary and mentioned it specifically.

'''
)
REFINE_PROMPT_SUMMARY = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_summary,
)
