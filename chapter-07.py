# LangChain Agents: YouTube Search and Transcript Analysis
# This example demonstrates how to create agents that can search YouTube and extract video transcripts

import os
import warnings
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Initialize the ChatOpenAI model with API key and configuration
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini", temperature=0)

from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Pull pre-built agent prompt template from LangChain Hub
prompt = hub.pull("hwchase17/openai-tools-agent")
messages = prompt.messages

from langchain_community.tools import YouTubeSearchTool

# Initialize YouTube search tool for finding videos
youtube_search_tool = YouTubeSearchTool()

tools = [youtube_search_tool]

# Create agent that can call tools and execute it with YouTube search capability
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Example 1: Basic YouTube search
result = agent_executor.invoke({"input": "Find some of the waveform podcast videos."})

print(result)

from langchain_core.tools import tool
from youtube_transcript_api import YouTubeTranscriptApi


@tool
def transcribe_video(video_url: str) -> str:
    """Extract transcript from YouTube video by parsing video ID from URL"""
    # Extract video ID from various YouTube URL formats
    transcript = YouTubeTranscriptApi.get_transcript(video_url.split("v=")[-1].split("&")[0])
    return transcript


# Expand tools to include both search and transcription capabilities
tools = [youtube_search_tool, transcribe_video]

# Create enhanced agent with both YouTube search and transcript extraction
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Example 2: Complex query requiring both search and transcript analysis
result = agent_executor.invoke({"input": "What topics does the waveform podcast cover?"})

# Display both the original query and the agent's response
print(result["input"])
print(result["output"])
