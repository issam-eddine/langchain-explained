# Retrieval-Augmented Generation (RAG) Implementation
# This script demonstrates how to build a RAG system using LangChain, Redis Vector Store, and OpenAI

import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize the ChatOpenAI model with API key and configuration
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini", temperature=0)

import redis

# Establish connection to Redis database for vector storage
r = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
    decode_responses=True,
    username=os.getenv("REDIS_USERNAME"),
    password=os.getenv("REDIS_PASSWORD"),
)

# Verify Redis connection is working
r.ping()

# Check how many documents are stored in Redis
keys = r.keys("*")
print(f"Total keys in Redis: {len(keys)}")

from langchain_openai import OpenAIEmbeddings

# Initialize embedding model for converting text to vectors
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")

from langchain_redis import RedisVectorStore

# Connect to existing Redis vector index containing YouTube podcast embeddings
rvs = RedisVectorStore.from_existing_index(
    embedding=embeddings,
    index_name="youtube-podcast-embeddings",
    redis_url=f"redis://default:{os.getenv('REDIS_PASSWORD')}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}",
)

# Create retriever to find top 10 most similar documents
retriever = rvs.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# Test retrieval with a sample query
results = retriever.invoke("business")

# Inspect retrieved documents
for i, e in enumerate(results):
    print()
    print(f"Document {i + 1}: {e.page_content}")

from langchain.prompts import ChatPromptTemplate

# Define prompt template for RAG - instructs LLM to answer based only on provided context
template = """
Answer the question based only on the following context: {context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser

# Parser to convert LLM response to plain string
output_parser = StrOutputParser()

# Build RAG chain: question -> retrieve relevant docs -> generate answer
chain = (
    {
        "context": (lambda x: x["question"]) | retriever,  # Retrieve docs based on question
        "question": (lambda x: x["question"]),  # Pass question through unchanged
    }
    | prompt  # Format context and question into prompt
    | llm  # Generate answer using LLM
    | output_parser  # Parse response to string
)

# Execute RAG pipeline with a business-related question
result = chain.invoke({"question": "What are the key points about business?"})

print(result)
