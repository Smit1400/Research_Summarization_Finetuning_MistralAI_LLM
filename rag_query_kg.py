from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
import os

os.environ["OPENAI_API_KEY"] = "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"

url = "neo4j+s://d1d5cc9b.databases.neo4j.io"
username="neo4j"
password="yul3ynAibcbizogTqGcbe_LAijgbqQf9FBKf4iK7lB8"
graph = Neo4jGraph(
    url=url,
    username=username,
    password=password
)

graph.refresh_schema()

cypher_chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=ChatOpenAI(temperature=0, model="gpt-4"),
    qa_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    validate_cypher=True, # Validate relationship directions
    verbose=True
)

print(cypher_chain.run("who wrote the paper?"))