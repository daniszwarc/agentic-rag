from typing import Any, Dict

from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

from dotenv import load_dotenv
load_dotenv()

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---Web Search---")
    question = state["question"]
    documents = state.get("documents", [])

    tavily_results = web_search_tool.invoke({"query": question})

    # Debug: print the keys to see what's available
    print("Keys in tavily_results:", tavily_results.keys())
    print("Full response:", tavily_results)

    joined_tavily_results = "\n".join(
        [result["content"] for result in tavily_results["results"]]
    )

    web_results = Document(page_content=joined_tavily_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

    

if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})