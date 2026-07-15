#   ==== Loading the Document ====
from chromadb import Embeddings # Used for generating embeddings for the documents
from langchain_community.document_loaders import PyPDFLoader # Used to load the PDF document to convert it later to document objects

doc_path = r"C:\Users\aea76\OneDrive\سطح المكتب\FirstRAG\a research to read for LLM security.pdf"
model = "llama3.2"

if doc_path: # Checking if the document path exist
    # Initializing the loader
    loader = PyPDFLoader(file_path=doc_path) # Loading the PDF document using PyPDFLoader and creating the document objects
    
    data = loader.load() #  The data is a collection of document objects, where each document object represents a page in the PDF. 
    
    print(f"Loaded {len(data)} document object from {doc_path}")
else:
    print("No document path provided.")

# Access content
print(type(data))
#content = data[0].page_content # Reading the first page of the document 
#print(content[:]) # printing the content of the first page


#   ==== Splitting the Documents into Chunks ====

from langchain_text_splitters import RecursiveCharacterTextSplitter # Used for splitting the documents into smaller chunks (Piece of text) for better processing and analysis
from langchain_community.vectorstores import Chroma # Chroma is a vector database that stores the embeddings of the documents for efficient retrieval and processing.

# Pipeline: 1) Split the text to small 'Chunks' | 2) Generate Embeddings for Text | 3) Store Embeddings in a Vector Database

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300) # chunk_size specifies the number of characters included in each chunk
chunks = text_splitter.split_documents(data) # Splitting the documents into chunks
# The chunks variable is a list of Document objects, where each Document object represents a chunk of the original document. Each chunk contains a portion of the text from the original document.

""" A chunk is a smaller piece of text that is easier to process and analyze. It allows for more efficient retrieval and processing of information from large documents.
 The chunk_size is set to 1200 characters, which means that each chunk will contain up to 1200 characters, symbols, numbers, and other text elements. 
 The chunk_overlap is set to 300 characters, which means that there will be an overlap of 300 characters between consecutive chunks. 
 This overlap helps to maintain context and continuity between chunks, ensuring that important information is not lost during the 
 splitting process."""
 
print("Splitting is Done....")

print(f"Num of Chunks: {len(chunks)}") # Printing the total number of chunks
# print(f"First Chunk: {chunks[0].page_content[:]}") # Printing the first chunk of the document
# print(f"Last Chunk: {chunks[-1].page_content[:]}") # Printing the last chunk of the document

"""
The chunk_size acts more as a maximum limit for the size of each chunk, rather than a strict requirement.

Whether they are equal depends on how the text is structured and which "splitter" you are using.

The 'RecursiveCharacterTextSplitter' is "smart." It doesn't just slice your text like a guillotine at exactly 1200 characters. 
Instead, it looks for logical breaks in the text. It tries to keep paragraphs and sentences together so the LLM gets coherent information. 
If a paragraph ends at 1150 characters, the splitter will likely cut it there to avoid breaking a sentence in half, 
rather than forcing it to reach exactly 1200.

"""



#   ==== Add to Vector Database ====
import ollama
from langchain_ollama import OllamaEmbeddings # Used for generating embeddings for the documents

ollama.pull("nomic-embed-text-v2-moe") # Pulling the embedding model

vector_db = Chroma.from_documents(
    documents = chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text-v2-moe"),
    collection_name="pdf_rag",
    persist_directory="./chroma_db")
print("Vector Database is created and persisted....")

""" 
Chroma db stores the embeddings of the documents for efficient retrieval and processing. 
It allows for fast similarity searches and retrieval of relevant chunks based on user queries. 
The persist_directory parameter specifies the directory where the vector database will be stored, allowing for persistence across sessions.
"""



#   ==== Retrieve from the Vector Database ====
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate # Used for creating prompts for the LLM
from langchain_core.output_parsers import StrOutputParser   # Used for parsing the output of the LLM into a string format

from langchain_ollama import ChatOllama # Used for interacting with the LLM

from langchain_core.runnables import RunnablePassthrough # Used for passing the input to the LLM without any modifications
from langchain_classic.retrievers.multi_query import MultiQueryRetriever # Used for retrieving relevant chunks from the vector database based on the query


llm = ChatOllama(
    model=model,
    temperature=0.8,
    top_p=0.95 
)

QUERY_PROMPT = PromptTemplate( # This is the prompt that will be sent to the RAG system when needed
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}"""
)


# MultiQueryRetriever is function that takes a user query and generates multiple alternative queries to retrieve relevant documents from a vector database.
retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), 
    llm, 
    prompt=QUERY_PROMPT
    )
# The retriever helps to improve the retrieval of relevant documents by providing different perspectives on the user query.

# LLM Prompt
template = """Answer the provided question based ONLY on the following context: {context}
Question: {question}
"""
# The template is used to create a prompt for the LLM, instructing it to answer the user question based solely on the retrieved context from the vector database. The {context} placeholder will be replaced with the relevant chunks retrieved by the retriever, and the {question} placeholder will be replaced with the user query.  

prompt = ChatPromptTemplate.from_template(template)

chain = (
     {"context": retriever, "question":RunnablePassthrough()} # The retriever retrieves relevant chunks from the vector database based on the user query, and the RunnablePassthrough passes the user query to the LLM without any modifications.
     | prompt
     | llm
     | StrOutputParser() # used to parse the output of the LLM into a string format
)

res = chain.invoke(input=("What is the main security issues that the paper tries to address?"))

print(res)

"""
The entire process involves loading a PDF document, splitting it into manageable chunks, storing those chunks in a vector database, 
and then using a retrieval-augmented generation (RAG) approach to answer user queries based on the content of the document. 
The RAG approach combines the power of large language models with efficient retrieval mechanisms to provide accurate and 
contextually relevant answers.
"""