from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.vectorstores import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_loaders import PyPDFLoader

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

loader = PyPDFLoader("data/UnderstandingDeepLearning_05_27_24_C.pdf")
document = loader.load()

data = " ".join(doc.page_content for doc in document)
# print(data)
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
text_splitter = SemanticChunker(embeddings)

docs = text_splitter.create_documents([data])

qdrant = Qdrant.from_documents(
    docs, 
    embeddings,
    path = "./db",
    collection_name = "udl"
)

retriever = qdrant.as_retriever()

#ReRanker

compressor = FlashrankRerank()
compression_retriever = ContextualCompressionRetriever(base_compressor = compressor, base_retriever=retriever)

query = "can you explain the transformer architecture?"

reranked_docs = compression_retriever.invoke(query)

for doc in reranked_docs:
    print(f"id: {doc.metadata['_id']}\n")
    print(f"text: {doc.page_content}\n")
    print(f"score: {doc.metadata['relevance_score']}")
    print("-"*80)
    print("\n\n")