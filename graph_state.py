# graph_state.py
from typing import TypedDict

class ResearchState(TypedDict):
    topic: str
    search_results: str
    summaries: str
    analysis: str
    final_report: str