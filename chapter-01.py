# This file demonstrates how to interact with the ChatOpenAI/ChatGroq model using LangChain.
# It includes examples of simple text generation, structured messages for context, 
# and structured output parsing using Pydantic models.

import os
import textwrap
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Initialize the ChatOpenAI model with API key and configuration
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini", temperature=0)
# llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-20b", temperature=0)

# Example 1: Simple text-to-text generation
# The most basic way to interact with an LLM - pass a string and get a response

result = llm.invoke("Explain LLMs.").content

print(result)

# Example 2: Using structured messages (System + Human)
# This approach gives you more control over the conversation context
# System messages set the AI's behavior and role
# Human messages represent user input

from langchain.schema import SystemMessage, HumanMessage

system_prompt = "You are a helpful assistant that can answer questions and help with tasks."
user_prompt = "What is PCA in machine learning?"

# Create a list of messages to establish context and ask a question
messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

result = llm.invoke(messages)
answer = result.content

print(answer)

# Example 3: Structured output parsing with Pydantic models
# Instead of getting raw text, we can force the LLM to return structured data
# This is useful for extracting specific information in a predictable format

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class FlightSearch(BaseModel):
    """Pydantic model to define the structure of flight search data we want to extract"""

    from_location: str = Field(description="Departure city or airport")
    to_location: str = Field(description="Destination city or airport")
    departure_date: str = Field(description="Departure date")


# Create a prompt template for consistent flight information extraction
flight_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at extracting flight search information from user queries. Extract the departure location, destination, and date from the user's message.",
        ),
        (
            "human",
            "{query}"
        ),
    ]
)

# Chain the prompt with the LLM and structured output parser
# This creates a pipeline: prompt -> LLM -> structured output
flight_extractor = flight_prompt | llm.with_structured_output(schema=FlightSearch)

# Test the extraction with a sample user query
user_query = "I would like to search for flights from Paris to London, on the 17th of February 2026."

# Extract structured flight information from the natural language query
flight_info = flight_extractor.invoke({"query": user_query})

print()
print(f"Dict: {dict(flight_info)}")

print()
print("Flight Search Extraction:")
print(f"From: {flight_info.from_location}")
print(f"To: {flight_info.to_location}")
print(f"Date: {flight_info.departure_date}")
