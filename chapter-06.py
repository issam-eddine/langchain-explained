# LangChain Tools Integration - YouTube Search Example
# This script demonstrates how to use LangChain tools with an LLM to search YouTube videos

import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize the ChatOpenAI model with API key and configuration
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini", temperature=0)

# Import and set up the YouTube search tool
from langchain_community.tools import YouTubeSearchTool

youtube_search_tool = YouTubeSearchTool()

# Test the YouTube search tool directly
youtube_search_tool.run("the waveform podcast")

# Bind the YouTube tool to the LLM so it can use the tool when needed
llm_with_tools = llm.bind_tools([youtube_search_tool])

# Invoke the LLM with a query - it will decide whether to use the YouTube tool
message = llm_with_tools.invoke("the waveform podcast")

# Display the tool calls made by the LLM
print(message.tool_calls)

# Create a chain that extracts the search query and passes it to the YouTube tool
chain = llm_with_tools | (lambda x: x.tool_calls[0]["args"]["query"]) | youtube_search_tool

# Execute the chain with a more specific request
result = chain.invoke("find 10 videos from the waveform podcast")

# Display the final results
print(result)
