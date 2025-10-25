# main.py
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from crewai import Task
from graph_state import ResearchState
from agents import (
    literature_search_agent,
    paper_summarizer_agent,
    theme_analyst_agent,
    report_writer_agent,
)
from tools import pinecone_search

# Load environment variables
load_dotenv()

# --- Define Node Functions ---

def literature_search_node(state: ResearchState):
    print("--- SEARCHING LITERATURE ---")
    task = Task(
        description=f"Find relevant academic literature on the topic: {state['topic']}. The search results must be comprehensive.",
        expected_output="A list of relevant excerpts from academic papers.",
        agent=literature_search_agent,
        tools=[pinecone_search]
    )
    result = literature_search_agent.execute_task(task)
    return {"search_results": result}

def summarize_papers_node(state: ResearchState):
    print("--- SUMMARIZING PAPERS ---")
    task = Task(
        description="For each excerpt found, create a concise summary highlighting the key findings.",
        expected_output="A compiled list of all summaries.",
        agent=paper_summarizer_agent,
        context={"search_results": state['search_results']}
    )
    result = paper_summarizer_agent.execute_task(task)
    return {"summaries": result}

def analyze_themes_node(state: ResearchState):
    print("--- ANALYZING THEMES ---")
    task = Task(
        description="Analyze the provided summaries to identify major themes, connections, and contrasting arguments.",
        expected_output="A report detailing the key themes and debates in the literature.",
        agent=theme_analyst_agent,
        context={"summaries": state['summaries']}
    )
    result = theme_analyst_agent.execute_task(task)
    return {"analysis": result}

def write_report_node(state: ResearchState):
    print("--- WRITING FINAL REPORT ---")
    task = Task(
        description="Write a comprehensive literature review report based on the provided summaries and theme analysis.",
        expected_output="A final, polished literature review document.",
        agent=report_writer_agent,
        context={"summaries": state['summaries'], "analysis": state['analysis']}
    )
    result = report_writer_agent.execute_task(task)
    return {"final_report": result}

# --- Build the Graph ---
workflow = StateGraph(ResearchState)
workflow.add_node("searcher", literature_search_node)
workflow.add_node("summarizer", summarize_papers_node)
workflow.add_node("analyst", analyze_themes_node)
workflow.add_node("writer", write_report_node)

workflow.set_entry_point("searcher")
workflow.add_edge("searcher", "summarizer")
workflow.add_edge("summarizer", "analyst")
workflow.add_edge("analyst", "writer")
workflow.add_edge("writer", END)

# Compile the graph into a runnable app
app = workflow.compile()

# The section below that automatically ran the crew has been removed.
# The `app` object is now ready to be imported by Streamlit.