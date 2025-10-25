# agents.py
import os
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import pinecone_search

# Explicitly configure the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key=os.environ.get("GEMINI_API_KEY")
)

# 1. Literature Search Agent
literature_search_agent = Agent(
    role='Literature Search Specialist',
    goal='Find the most relevant academic papers and excerpts for a given research topic using the Pinecone index.',
    backstory=(
        "You are an expert at navigating complex academic databases. Your sole mission is to use the "
        "provided search tool to find the most precise and relevant information from the knowledge base."
    ),
    tools=[pinecone_search],
    llm=llm,
    verbose=True,
)

# 2. Paper Summarizer Agent
paper_summarizer_agent = Agent(
    role='Expert Paper Summarizer',
    goal='Read the retrieved academic excerpts and produce clear, concise summaries for each.',
    backstory=(
        "You have a unique talent for distilling dense, technical academic texts into easy-to-understand summaries. "
        "You focus on the key findings and methodologies of each paper."
    ),
    tools=[],
    llm=llm,
    verbose=True,
)

# 3. Theme Analyst Agent
theme_analyst_agent = Agent(
    role='Theme and Connection Analyst',
    goal='Analyze a collection of paper summaries to identify overarching themes, contrasting arguments, and connections.',
    backstory=(
        "You are a master of synthesis, able to see the forest for the trees. You take individual summaries "
        "and weave them together, identifying the key conversations and debates happening within the literature."
    ),
    tools=[],
    llm=llm,
    verbose=True,
)
# 4. Report Writer Agent
report_writer_agent = Agent(
    role='Academic Report Writer',
    goal='Write a final, well-structured literature review report based on the identified themes and summaries.',
    backstory=(
        "You are a skilled academic writer, known for your clarity and ability to structure complex information. "
        "Your task is to take the analyzed themes and summaries and compose a final literature review report."
    ),
    tools=[],
    llm=llm,
    verbose=True,
)