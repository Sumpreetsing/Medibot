import streamlit as st
import os
import warnings
warnings.filterwarnings("ignore")

# Import other libraries after streamlit to avoid conflicts
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Set page config first
st.set_page_config(
    page_title="Ask Chatbot",
    page_icon="🤖",
    layout="wide"
)

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Load the FAISS vector store with embeddings"""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Use 'cuda' if you have GPU
        )
        db = FAISS.load_local(
            DB_FAISS_PATH, 
            embedding_model, 
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template):
    """Create a custom prompt template"""
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )
    return prompt

@st.cache_resource
def load_llm(huggingface_repo_id, hf_token):
    """Load the Hugging Face LLM with proper authentication"""
    if not hf_token:
        raise ValueError("Hugging Face token is required. Please set the HF_TOKEN environment variable.")
    
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            huggingfacehub_api_token=hf_token,
            temperature=0.5,
            max_new_tokens=512,
            top_p=0.95,
            repetition_penalty=1.15
        )
        return llm
    except Exception as e:
        raise Exception(f"Failed to load LLM: {str(e)}")

def check_hf_token():
    """Check if Hugging Face token is available"""
    # Check multiple sources for the token
    token = (
        os.environ.get("HF_TOKEN") or 
        os.environ.get("HUGGINGFACE_HUB_TOKEN") or 
        st.secrets.get("HF_TOKEN", None) if hasattr(st, 'secrets') else None
    )
    
    # If still no token, try to get from streamlit secrets directly
    if not token:
        try:
            token = st.secrets["HF_TOKEN"]
        except:
            pass
    
    return token

def main():
    st.title("🤖 Ask Chatbot")
    st.markdown("---")
    
    # Check for HF token first
    hf_token = check_hf_token()
    if not hf_token:
        st.error("""
        🔑 **Hugging Face Token Required!**
        
        Please set your Hugging Face token in one of these ways:
        
        **Method 1: Environment Variable**
        ```bash
        export HF_TOKEN="your_token_here"
        ```
        
        **Method 2: Streamlit Secrets**
        Add to `.streamlit/secrets.toml`:
        ```toml
        HF_TOKEN = "your_token_here"
        ```
        
        **Get your token from:** https://huggingface.co/settings/tokens
        """)
        st.stop()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    prompt = st.chat_input("Enter your question here...")
    
    if prompt:
        # Display user message
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        # Custom prompt template
        CUSTOM_PROMPT_TEMPLATE = """
        Use the following context to answer the question. Be concise and accurate.
        If you don't know the answer, just say you don't know.

        Context: {context}
        Question: {question}

        Answer:
        """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        
        try:
            with st.spinner("🔍 Searching knowledge base..."):
                # Load vector store
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("❌ Failed to load the vector store. Please check if the vectorstore directory exists.")
                    st.stop()
                
                # Load LLM
                llm = load_llm(HUGGINGFACE_REPO_ID, hf_token)
                
                # Create QA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )
                
                # Get response
                response = qa_chain.invoke({'query': prompt})
                
                result = response["result"]
                source_documents = response["source_documents"]
                
                # Format response with sources
                if source_documents:
                    sources_info = "\n\n📚 **Sources:**\n"
                    for i, doc in enumerate(source_documents, 1):
                        # Get page content preview (first 100 chars)
                        content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                        sources_info += f"{i}. {content_preview}\n"
                    
                    result_to_show = result + sources_info
                else:
                    result_to_show = result
                
                # Display assistant response
                with st.chat_message('assistant'):
                    st.markdown(result_to_show)
                
                st.session_state.messages.append({
                    'role': 'assistant', 
                    'content': result_to_show
                })
                
        except Exception as e:
            error_message = f"❌ **Error:** {str(e)}"
            if "401" in str(e) or "Unauthorized" in str(e):
                error_message += "\n\n🔑 **Authentication Issue:** Please check your Hugging Face token."
            elif "load_local" in str(e):
                error_message += f"\n\n📁 **Vector Store Issue:** Please ensure the vectorstore exists at: `{DB_FAISS_PATH}`"
            
            st.error(error_message)
            
            # Add error to chat history
            with st.chat_message('assistant'):
                st.error("Sorry, I encountered an error while processing your request.")

if __name__ == "__main__":
    main()