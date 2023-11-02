from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
)

if __name__ == "__main__":
    loader = DirectoryLoader("research_papers/", glob="*.pdf", loader_cls=PyPDFLoader)
    load_data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    documents = text_splitter.split_documents(load_data)

    neo4j_vector = Neo4jVector.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        url=os.environ["NEO4J_URI"],
        username=os.environ["NEO4J_USERNAME"],
        password=os.environ["NEO4J_PASSWORD"],
    )

    # graph.refresh_schema
    # print(graph.schema)

    query = "Which models did the authors of the paper proposed, please tell in detail in pointer"
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=neo4j_vector.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True
                                       )

    print(
        qa_chain({"query": query})['result']
    )
