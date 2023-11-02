from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

print(
    os.environ["NEO4J_URI"], os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]
)

graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
)
# graph.refresh_schema()

# cypher_chain = GraphCypherQAChain.from_llm(
#     graph=graph,
#     cypher_llm=ChatOpenAI(temperature=0, model="gpt-4"),
#     qa_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
#     validate_cypher=True, # Validate relationship directions
#     verbose=True
# )

# print(cypher_chain.run("Which model is being used and why?"))
