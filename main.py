from langchain.evaluation import load_evaluator
from fastapi import FastAPI, Response, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
import requests
from datetime import datetime
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
import json

app = FastAPI()



# Set folder paths
UPLOADS_FOLDER = os.path.abspath("uploads")
DB_FOLDER = os.path.abspath("db")
BACKUP_FOLDER = os.path.abspath("db_backup")

# Initialize embeddings and text splitter
embeddings = OllamaEmbeddings(model="nomic-embed-text")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


# Create necessary directories if they don't exist
os.makedirs(UPLOADS_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)
os.makedirs(BACKUP_FOLDER, exist_ok=True)

@app.get("/")
def home():
    """
    Home endpoint that backs up and deletes the Chroma database on load.
    """
    try:
        # Backup the database
        if os.path.exists(DB_FOLDER):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(BACKUP_FOLDER, f"db_backup_{timestamp}")
            shutil.copytree(DB_FOLDER, backup_path)
            print(f"Database backed up at: {backup_path}")

        # Delete the original database
        if os.path.exists(DB_FOLDER):
            shutil.rmtree(DB_FOLDER)
            os.makedirs(DB_FOLDER)  # Recreate an empty DB_FOLDER
            print("Original database deleted and folder reset.")
        
        return {"message": "Welcome! Database has been backed up and reset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during backup and delete: {str(e)}")

@app.post('/pdfs/')
async def upload_pdf(file: UploadFile):
   try:
        # Validate file extension
        if not file.filename.lower().endswith(('.pdf')):
            raise HTTPException(status_code = 400,detail="Invalid file type. Only .pdf files are allowed.")

        # Save the uploaded file
        file_contents = await file.read()
        file_path = os.path.join(UPLOADS_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(file_contents)

        # Process the uploaded file
        loader = PDFPlumberLoader(file_path)
        docs = loader.load_and_split()
        chunks = splitter.split_documents(docs)

        # Update vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_FOLDER
        )

        return {
            "filename": file.filename,
            "file_path": file_path,
            "size": len(file_contents),
            "docs": len(docs),
            "chunks": len(chunks)
        }
   except HTTPException as e:
        raise e
   except Exception as e:
       pass
 
@app.post('/docs/')
async def upload_docs(file: UploadFile):
    try:
        # Validate file extension
        if not file.filename.lower().endswith(('.docx')):
            raise HTTPException(status_code=400, detail="Invalid file type. Only .docx files are allowed.")

        # Save the uploaded file
        file_contents = await file.read()
        file_path = os.path.join(UPLOADS_FOLDER, file.filename)
        os.makedirs(UPLOADS_FOLDER, exist_ok=True)  # Ensure the upload directory exists
        with open(file_path, "wb") as f:
            f.write(file_contents)

        # Process the uploaded file
        loader = Docx2txtLoader(file_path)
        docs = loader.load_and_split()
        chunks = splitter.split_documents(docs)

        # Update vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_FOLDER
        )

        return {
            "filename": file.filename,
            "file_path": file_path,
            "size": len(file_contents),
            "docs": len(docs),
            "chunks": len(chunks)
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        pass



@app.post("/web_scrape")
async def web_scrape(url: str):
    try:
        # Initialize the loader (asynchronous)
        loader = AsyncHtmlLoader(url)
        docs =  loader.load()  # Use await here

        print("docs:", docs)

        # Check if documents are loaded successfully
        if not docs:
            raise HTTPException(status_code=400, detail="No content found at the provided URL.")

        # Transform HTML to text
        bs2 = BeautifulSoupTransformer()
        text = bs2.transform_documents(docs, tags_to_extract=["p", "li", "div", "a", "h1", "h2", "h3", "h4", "h5", "h6"])
        print("text:", text)

        chunks = splitter.split_documents(text)
        print("chunks:", chunks)

        # Update the vector store
        for chunk in chunks:
            vector_store = Chroma.from_documents([chunk], embeddings, persist_directory=DB_FOLDER)

        # Calculate the total text length
        total_text_length = sum(len(chunk.page_content) for chunk in chunks)
        print("total_text_length:", total_text_length)

        return JSONResponse(
            content={
                "message": "Web scraping completed successfully.",
                "url": url,
                "size": total_text_length,
                "total_chunks": len(chunks),
            },
            status_code=200
        )

    except HTTPException as e:
        raise e

    except Exception as e:
        print("Error:", e)
        pass

    

@app.post('/ai_chat')
def ai_Post(prompt: str):
    # Load the vector store
    vector_store = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)

    # Perform similarity search
    docs = vector_store.similarity_search(prompt,k=4)
    # Define prompt template
    prompt_template = PromptTemplate(
    template=(
        "You are a knowledgeable assistant. "
        "Answer the question based on the provided context:\n\n"
        "Context: {context}\n"
        "Question: {question}\n"
        "Answer: [Provide a detailed response with no paraphrasing.]\n"
        "Reliability: [Provide reliability by describing where the information comes from.]"
    ),
    input_variables=["context", "question"]
)


    # Initialize LLM and QA chain
    llm = ChatOllama(model="llama3.2", temperature=0.5)
    chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)

    #Invoking_chain 
    response = chain.invoke({
    "context": docs,
    "question": prompt
})
    answer = response
    
    return {"response": answer}