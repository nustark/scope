from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI
from langchain.chains import RetrievalQA
import pinecone
import os

load_dotenv()

loader = DirectoryLoader(
    './files/mech',
    glob='**/*.pdf',
    show_progress=True
)
docs = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)
docs_split = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

doc_db = Pinecone.from_documents(
    docs_split,
    embeddings,
    index_name='llm-test'
)

# search from loaded documents using cosine similarity metric
# query = ""
# search_docs = doc_db.similarity_search(query)

llm = OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=doc_db.as_retriever(),
)

query = "What are some immediate and gradual changes I can make to my desk ergonomics to alleviate carpal tunnel?"
result = qa.run(query)

print(result)
