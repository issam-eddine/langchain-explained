# Document Splitting and Vector Retrieval with Redis
# This script demonstrates how to load YouTube transcripts, split them into chunks,
# embed them using OpenAI, store them in Redis, and perform similarity searches

import os
import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize the ChatOpenAI model with API key and configuration
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini", temperature=0)

from langchain_community.document_loaders import YoutubeLoader

# Configure and load YouTube video transcript
# Using French language setting for transcript extraction
youtube_url = "https://www.youtube.com/watch?v=PUUg6JCV9dg"
loader = YoutubeLoader.from_youtube_url(youtube_url=youtube_url, add_video_info=False, language="fr")

# Extract the transcript as LangChain documents
documents = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure text splitter to break documents into manageable chunks
# Small chunk size (100 chars) with overlap to maintain context between chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,  # 100 characters per chunk
    chunk_overlap=20,  # 20 characters overlap between chunks
    length_function=len,  # Use the length of the text as the length function
    is_separator_regex=False,  # False means use the default separator regex
)

# Split the loaded documents into smaller chunks for better retrieval
documents_split = text_splitter.split_documents(documents)

print(f"Number of documents: {len(documents_split)}")

# Display first few chunks to inspect the splitting results
print(documents_split[0])
print(documents_split[1])
print(documents_split[2])

import redis

# Establish connection to Redis database using environment credentials
r = redis.Redis(
    host=os.getenv("REDIS_HOST"),
    port=os.getenv("REDIS_PORT"),
    decode_responses=True,
    username=os.getenv("REDIS_USERNAME"),
    password=os.getenv("REDIS_PASSWORD"),
)

# Verify Redis connection is working
r.ping()

# Clear any existing data in the database for a fresh start
r.flushdb()

from langchain_openai import OpenAIEmbeddings

# Initialize OpenAI embeddings model for converting text to vectors
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model="text-embedding-3-small")

from langchain_redis import RedisVectorStore

# Select a random sample of 256 documents to avoid overwhelming the system
# This is useful for testing purposes when working with large datasets
sample_indices = np.random.choice(range(len(documents_split)), size=256, replace=False)
 
# Create Redis vector store and populate it with embedded document chunks
# This process converts text chunks to vectors and stores them in Redis for fast retrieval
rvs = RedisVectorStore.from_documents(
    documents=[documents_split[x] for x in sample_indices],  # Use the random sample of 64 documents
    embedding=embeddings,
    index_name="youtube-podcast-embeddings",
    redis_url=f"redis://default:{os.getenv('REDIS_PASSWORD')}@{os.getenv('REDIS_HOST')}:{os.getenv('REDIS_PORT')}",
)

# Verify that documents were successfully stored in Redis
keys = r.keys("*")
print(f"Total keys in Redis: {len(keys)}")

# Configure retriever to find the most semantically similar documents
# k = 10 means it will return the top 10 most similar chunks
retriever = rvs.as_retriever(search_type="similarity", search_kwargs={"k": 8})

# Perform a similarity search for business-related content
results = retriever.invoke("business")

# Display the retrieved documents and their content
for i, e in enumerate(results):
    print()
    print(f"Document {i + 1}: {e.page_content}")
