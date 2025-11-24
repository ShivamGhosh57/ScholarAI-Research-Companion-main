# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict
import database  # Ensure database.py is in the same folder

# Core AI Libraries
from langgraph.graph import StateGraph, END
from crewai import Agent, Task
from crewai_tools import BaseTool

# LLM and Tool Libraries
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- 1. Initialization & Secret Management ---
st.set_page_config(page_title="AI Academic Research Assistant", layout="wide")
load_dotenv()
database.init_db()

# Helper function to get secrets safely (Local .env vs Streamlit Cloud Secrets)
def get_secret(key):
    if key in os.environ:
        return os.environ[key]
    elif key in st.secrets:
        return st.secrets[key]
    return None

# Load keys
groq_api_key = get_secret("GROQ_API_KEY")
pinecone_api_key = get_secret("PINECONE_API_KEY")

# Validate keys
if not groq_api_key or not pinecone_api_key:
    st.error("🚨 Missing API Keys! Please check your .env file (local) or Streamlit Secrets (cloud).")
    st.stop()

# Set keys to environment variables so libraries like Pinecone/LangChain can find them automatically
os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key

# --- 2. Define Tools ---
class PineconeSearchTool(BaseTool):
    name: str = "Pinecone Academic Search"
    description: str = "Searches your private library of academic papers in Pinecone."
    
    def _run(self, query: str) -> str:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = PineconeVectorStore(index_name="academic-research", embedding=embeddings)
        docs = vectorstore.similarity_search(query, k=5)
        return "\n---\n".join([doc.page_content for doc in docs])

pinecone_search_tool = PineconeSearchTool()

# --- 3. Define Agents ---
# Initialize LLM using the retrieved key
llm = ChatGroq(
    model="llama3-8b-8192",
    groq_api_key=groq_api_key
)

literature_search_agent = Agent(
    role='Literature Search Specialist',
    goal='Find relevant academic papers for a research topic.',
    backstory="You are an expert at navigating academic databases.",
    tools=[pinecone_search_tool],
    llm=llm,
    verbose=True
)

paper_summarizer_agent = Agent(
    role='Expert Paper Summarizer',
    goal='Produce clear, concise summaries of academic excerpts.',
    backstory="You distill dense academic texts into understandable summaries.",
    tools=[],
    llm=llm,
    verbose=True
)

theme_analyst_agent = Agent(
    role='Theme and Connection Analyst',
    goal='Analyze paper summaries to identify overarching themes and connections.',
    backstory="You are a master of synthesis, identifying key debates in the literature.",
    tools=[],
    llm=llm,
    verbose=True
)

report_writer_agent = Agent(
    role='Academic Report Writer',
    goal='Write a final, well-structured literature review report.',
    backstory="You are a skilled academic writer.",
    tools=[],
    llm=llm,
    verbose=True
)

# --- 4. Define Graph State ---
class ResearchState(TypedDict):
    topic: str
    search_results: str
    summaries: str
    analysis: str
    final_report: str

# --- 5. Define Node Functions ---
def literature_search_node(state: ResearchState):
    task = Task(
        description=f"Find relevant academic literature on the topic: {state['topic']}.",
        expected_output="A list of relevant excerpts.",
        agent=literature_search_agent
    )
    result = literature_search_agent.execute_task(task)
    return {"search_results": result}

def summarize_papers_node(state: ResearchState):
    task = Task(
        description=f"Create a concise summary for the following search results:\n\n{state['search_results']}",
        expected_output="A compiled list of all summaries.",
        agent=paper_summarizer_agent
    )
    result = paper_summarizer_agent.execute_task(task)
    return {"summaries": result}

def analyze_themes_node(state: ResearchState):
    task = Task(
        description=f"Analyze the following summaries to identify major themes:\n\n{state['summaries']}",
        expected_output="A report detailing key themes.",
        agent=theme_analyst_agent
    )
    result = theme_analyst_agent.execute_task(task)
    return {"analysis": result}

def write_report_node(state: ResearchState):
    task = Task(
        description=f"Write a comprehensive literature review report based on the summaries and analysis:\n\nSUMMARIES:\n{state['summaries']}\n\nANALYSIS:\n{state['analysis']}",
        expected_output="A final, polished literature review document.",
        agent=report_writer_agent
    )
    result = report_writer_agent.execute_task(task)
    return {"final_report": result}

# --- 6. Build Workflow ---
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

app = workflow.compile()

# --- 7. RAG Chain Logic ---
def create_rag_chain(report_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(report_text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    template = "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}"
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt 
        | llm 
        | StrOutputParser()
    )
    return rag_chain

# --- 8. Streamlit Interface ---

# Initialize session state
if "selected_conversation_id" not in st.session_state:
    st.session_state.selected_conversation_id = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# --- Sidebar ---
with st.sidebar:
    st.title("📚 Assistant Menu")
    
    if st.button("➕ New Research"):
        st.session_state.selected_conversation_id = None
        st.session_state.rag_chain = None
        st.rerun()

    st.subheader("Recent Conversations")
    conversations = database.get_conversations()
    if not conversations:
        st.info("No history yet.")
        
    for convo in conversations:
        # Display conversations as buttons
        if st.button(convo[1], key=f"convo_{convo[0]}"):
            st.session_state.selected_conversation_id = convo[0]
            # Load context for RAG
            messages = database.get_messages(convo[0])
            if messages:
                # Assume first assistant message is the report
                report_text = messages[0]['content']
                st.session_state.rag_chain = create_rag_chain(report_text)
            else:
                st.session_state.rag_chain = None
            st.rerun()

# --- Main Area ---
st.title("AI Academic Research Assistant 🧠")

if st.session_state.selected_conversation_id is None:
    # [NEW RESEARCH MODE]
    st.write("Enter a research topic below to begin.")
    topic = st.text_input("Research Topic:", placeholder="e.g., The Impact of Large Language Models on Software Development")

    if st.button("Start Research"):
        if not topic:
            st.warning("Please enter a research topic.")
        else:
            with st.spinner("The AI Crew is starting the research..."):
                try:
                    inputs = {"topic": topic}
                    final_state = app.invoke(inputs)
                    final_report = final_state.get('final_report', 'No report was generated.')
                    
                    if final_report:
                        # Save to DB
                        new_convo_id = database.create_conversation(topic)
                        database.add_message(new_convo_id, "assistant", final_report)
                        
                        # Update Session State
                        st.session_state.selected_conversation_id = new_convo_id
                        st.session_state.rag_chain = create_rag_chain(final_report)
                        st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {e}")

else:
    # [CHAT MODE]
    messages = database.get_messages(st.session_state.selected_conversation_id)
    
    if messages:
        # Display the Report (First message)
        st.markdown("### 📑 Research Report")
        st.markdown(messages[0]['content'])
        st.markdown("---")
        st.subheader("💬 Chat with this Report")

    # Display Chat History (Excluding the first report message)
    chat_history = messages[1:]
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a follow-up question..."):
        # 1. Save User Message
        database.add_message(st.session_state.selected_conversation_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate & Save Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.rag_chain:
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(response)
                    database.add_message(st.session_state.selected_conversation_id, "assistant", response)
                else:
                    st.error("Chat context missing. Please restart research.")
