from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["NEO4J_URI"] = os.getenv('NEO4J_URI')
os.environ["NEO4J_USERNAME"] = os.getenv('NEO4J_USERNAME')
os.environ["NEO4J_PASSWORD"] = os.getenv('NEO4J_PASSWORD')

print(os.environ["NEO4J_URI"], os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])

graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"]
)

if __name__ == '__main__':
    
    loader = DirectoryLoader("research_papers/", glob="*.pdf", loader_cls=PyPDFLoader)
    load_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = text_splitter.split_documents(load_data)
    
    neo4j_vector = Neo4jVector.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"]
    )

    graph.refresh_schema
    print(graph.schema)
    
    query = "how rnn will help in text summarization"
    vector_results = neo4j_vector.similarity_search(query, k=2)
    vector_result = vector_results[0].page_content

    cypher_chain = GraphCypherQAChain.from_llm(
        graph=graph,
        cypher_llm=ChatOpenAI(temperature=0, model="gpt-4"),
        qa_llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        validate_cypher=True, # Validate relationship directions
        verbose=True
    )

    print(cypher_chain.run(query))