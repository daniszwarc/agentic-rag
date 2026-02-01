from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource"""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ..., 
        description="Given the user question, chose to route it to a web search or a vectorstore.",
    )

llm = ChatOpenAI(model="gpt-5.2", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert routing a user question to a vectorestore or web search.
The vectorestore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For everything else, use web-search.
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
