# Working with Prompts, Chains, and Document Loaders
# This script demonstrates core LangChain concepts including prompt templates,
# chains for connecting components, and document loaders for external data sources

import os
import textwrap
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Initialize the ChatOpenAI model with API key and configuration
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini", temperature=0)

# Define a prompt template with a placeholder for dynamic input
# This template instructs the AI to explain topics in simple English
prompt_template = """
Given the following topic, explain it in a way that is easy to understand. It the topic is in a language other than English, you should still answer in English.
Topic: {topic}
"""

# Create a PromptTemplate object that can format the template with variables
# This allows us to reuse the same prompt structure with different inputs
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["topic"],
)

# Test the prompt formatting with a sample topic
print(prompt.format(topic="machine learning"))

# Create a chain by connecting the prompt and language model using the pipe operator
# Chains allow us to sequence operations: prompt formatting -> LLM processing
chain = prompt | llm

# Execute the chain with input data
# The chain will format the prompt and send it to the LLM for processing
result = chain.invoke({"topic": "machine learning"})

print(result.content)

# Demonstrate document loading from external sources
# This example shows how to load and process YouTube video transcripts

from langchain_community.document_loaders import YoutubeLoader

# Initialize YouTube loader with a specific video URL
# Setting add_video_info=False to focus only on transcript content
# Language set to French as an example of multilingual content processing
youtube_url = "https://www.youtube.com/watch?v=PUUg6JCV9dg"
loader = YoutubeLoader.from_youtube_url(youtube_url=youtube_url, add_video_info=False, language="fr")

# Extract transcript content as LangChain Document objects
# Each document contains page_content (transcript) and metadata
documents = loader.load()

# Reuse the same chain to process each document segment
# This demonstrates how chains can be applied to multiple data sources
summary_chain = prompt | llm

# Process each document segment through the chain
# The AI will explain the French content in English as specified in our prompt
for doc in documents:
    print(doc.metadata)
    result = summary_chain.invoke({"topic": doc.page_content})
    print(f"Summary for video segment: {result.content}")
