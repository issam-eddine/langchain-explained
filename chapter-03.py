# LangChain Expression Language (LCEL) and Runnables
# LCEL is a declarative way to compose chains using Runnables -
# the fundamental building blocks of LangChain that can be invoked, streamed, and composed together.
# Key Runnable types: RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

import os
import textwrap
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize the ChatOpenAI model with API key and configuration
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini", temperature=0)

# Define a prompt template for multi-level summarization
summarize_prompt_template = """
You are a helpful assistant that explains at different levels of expertise:

<text>
{text}
</text>

Summarize the text above:
1. as if I am a high school student
3. as if I am a PhD student in mathematics
"""

summarize_prompt = PromptTemplate(template=summarize_prompt_template, input_variables=["text"])

# ===== BASIC CHAIN COMPOSITION =====
# Create a simple chain using the pipe operator (|) to connect components

output_parser = StrOutputParser()

# Chain: prompt -> llm -> parser (this creates a RunnableSequence)
chain = summarize_prompt | llm | output_parser

# Test the basic chain
result = chain.invoke({"text": "Principal Component Analysis."})
print(result)

print(f"Chain type: {type(chain)}")

# ===== USING RUNNABLE LAMBDA =====
# RunnableLambda allows you to wrap custom functions as Runnables

from langchain_core.runnables import RunnableLambda

summarize_chain = summarize_prompt | llm | output_parser

# Wrap a lambda function to make it a Runnable component
length_lambda = RunnableLambda(lambda summary: f"Summary length: {len(summary)} characters")

# Extend the chain with our custom lambda
lambda_chain = summarize_chain | length_lambda

# Uncomment to test the lambda chain
# result = lambda_chain.invoke({'text': "Principal Component Analysis."})
# print(result)

print(f"Lambda step type: {type(lambda_chain.steps[-1])}")

# ===== IMPLICIT RUNNABLE LAMBDA CONVERSION =====
# You can use functions directly in chains without explicit RunnableLambda wrapping
# LangChain automatically converts them

lambda_chain = summarize_chain | (lambda x: f"Summary length: {len(x)} characters")

result = lambda_chain.invoke({"text": "Principal Component Analysis."})
print(f"Result: {result}")

print(f"Auto-converted lambda type: {type(lambda_chain.steps[-1])}")

# ===== RUNNABLE PASSTHROUGH =====
# RunnablePassthrough allows data to flow through unchanged or with additions

from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda

length_lambda = RunnableLambda(lambda summary: f"Summary length: {len(summary)} characters")
summarize_chain = summarize_prompt | llm | output_parser

# Simple passthrough - data flows unchanged to the next step
passthrough = RunnablePassthrough()
placeholder_chain = summarize_chain | passthrough | length_lambda

# ===== PASSTHROUGH WITH ASSIGNMENT =====
# More advanced: add new fields to the data while preserving original

# Wrap the summary in a dictionary structure
wrap_summary_lambda = RunnableLambda(lambda summary: {"summary": summary})

# RunnablePassthrough.assign() adds new fields to existing dictionary
assign_passthrough = RunnablePassthrough.assign(length=lambda x: len(x["summary"]))

# Complete chain: text -> summary -> wrap in dict -> add length field
summarize_chain = summarize_prompt | llm | output_parser | wrap_summary_lambda
assign_chain = summarize_chain | assign_passthrough

# This returns a dict with both 'summary' and 'length' keys
result = assign_chain.invoke({"text": "What is PCA?"})
print(f"Assign result: {result}")

print(f"Passthrough step type: {type(assign_chain.steps[-1])}")

# ===== RUNNABLE PARALLEL =====
# Execute multiple operations on the same input simultaneously

from langchain_core.runnables import RunnableParallel

# Create the base summarization chain
summarize_chain = summarize_prompt | llm | output_parser

# RunnableParallel runs multiple functions on the same input
# Returns a dictionary with results from each parallel operation
parallel_runnable = RunnableParallel(summary=lambda x: x, length=lambda x: len(x))

# Chain: text -> summary -> parallel processing (identity + length calculation)
parallel_chain = summarize_chain | parallel_runnable

result = parallel_chain.invoke({"text": "Principal Component Analysis."})
print(f"Parallel result: {result}")

print(f"Parallel step type: {type(parallel_chain.steps[-1])}")
