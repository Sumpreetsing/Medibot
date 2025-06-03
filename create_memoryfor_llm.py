from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# load raw files
data_path='data/'
def load_pdf_files(data):
    loader=DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents
documents=load_pdf_files(data_path)
# print("Length of file :",len(documents))

# creating chinks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks=create_chunks(extracted_data=documents)
# print("length of chunks :",len(text_chunks))

# creating vector embeddings
def get_embedding_model():
    embeding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeding_model

embedding_model=get_embedding_model()

#store embeddings in FAISS
db_faiss_path="vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(db_faiss_path)
