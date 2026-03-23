import os

from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector


load_dotenv()


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

splitted_documents = text_splitter.split_documents(docs_list)

# Inicializa el modelo de embeddings configura la instancia con el modelo "text-embedding-3-large" 
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"), model="text-embedding-3-large")


print("ingesting...")
#PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])
vectorstore = PGVector.from_documents(  # metodo para crear una base de datos vectorial a partir de documentos.
    documents=splitted_documents,  # toma una lista de documentos de texto spliteados
    embedding=embeddings,  # genera representaciones vectoriales los textos spliteados, los documents
    connection="postgresql+psycopg://postgres:postgres@localhost:5432/vectordb",
    collection_name='ia_docs',  #  Define el nombre de la tabla/colección donde se guardan.
    use_jsonb=True,  # Almacena los metadatos de los documentos en formato jsonb de PostgreSQL para búsquedas más rápidas y eficientes. 
)
print("finish")

# devuelve un retriever que consultará el vectorstore y traerá los 3 documentos más relevantes para una consulta.
#retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever = vectorstore.as_retriever()  # es el cliente para consultar a la base de datos
# este retriever se puede invocar con "retriever.invoke(question)"

# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="rag-chroma",
#     embedding=OpenAIEmbeddings(),
#     persist_directory="./.chroma",
# )

# retriever = Chroma(
#     collection_name="rag-chroma",
#     persist_directory="./.chroma",
#     embedding_function=OpenAIEmbeddings(),
# ).as_retriever()
