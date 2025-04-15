import os
import ssl
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain
import certifi

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
import tempfile

load_dotenv()

# Set SSL certificate path
os.environ['SSL_CERT_FILE'] = certifi.where()

class ResearchAssistantLangChain:

    def __init__(self):
        """Initialize the Research Assistant with Groq models"""
        try:
            # Initialize Groq with SSL verification
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001",temperature=0,max_tokens=None,timeout=None,max_retries=2)
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.vector_db = None
            # Create a temporary directory for Chroma
            self.persist_directory = tempfile.mkdtemp()
            print(f"Created temporary Chroma directory: {self.persist_directory}")
        except Exception as e:
            print(f"Error initializing ResearchAssistantLangChain: {str(e)}")
            raise
    

    def load_urls(self, urls: List[str]) -> List[Document]:
        all_documents = []
    
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                data = loader.load()
                
                # Split documents into chunks
                split_documents = self.text_splitter.split_documents(data)
                all_documents.extend(split_documents)
                
                print(f"Successfully loaded and processed {url}")
            except Exception as e:
                print(f"Error loading {url}: {e}")
        
        return all_documents


    def create_vector_store(self, documents: List[Document]):


        # Use FAISS instead of Chroma to avoid tenant issues
        try:
            from langchain_community.vectorstores import FAISS
            
            self.vector_db = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            print(f"Created FAISS vector store with {len(documents)} documents")
        except Exception as e:
            print(f"Error creating FAISS vector store: {e}")
            # Fallback to in-memory Chroma
            try:
                self.vector_db = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=None  # In-memory only
                )
                print("Created in-memory Chroma vector store")
            except Exception as e2:
                print(f"Error creating in-memory Chroma: {e2}")
                raise ValueError(f"Failed to create vector store: {e2}")
            

    def query_data(self, query: str, num_results: int = 5) -> Dict[str, Any]:

        if not self.vector_db:
            raise ValueError("Vector store not initialized. Please load documents first.")
        
        retriever = self.vector_db.as_retriever(search_kwargs={"k": num_results})


        prompt_text = """
        Answer the following question based only on the provided context:
        
        <context>
        {context}
        </context>
        
        Question: {input}
        """


        # Create prompt template
        prompt = ChatPromptTemplate.from_template(prompt_text)
        
        # Create document chain
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Execute the chain
        response = retrieval_chain.invoke({"input": query})

        return {
            "answer": response["answer"],
            "source_documents": response["context"]
        }
    

    def summarize_document(self, document: str) -> str:

    

        message = HumanMessage(content=f"Summarize the following document in a concise but comprehensive manner:\n\n{document}")
        
        # Invoke the model directly
        response = self.llm.invoke([message])
        return response.content