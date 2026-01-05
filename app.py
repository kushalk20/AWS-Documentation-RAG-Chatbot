import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os


# Page config
st.set_page_config(
    page_title="AWS RAG Chatbot",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

VECTORSTORE_DIR = "vectorstore"

# Custom CSS
st.markdown("""
<style>
    /* Sticky Header */
    .header-section {
        position: sticky;
        top: 0;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        z-index: 1000;
        padding: 1rem;
        border-radius: 0 0 1rem 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Main header text */
    .main-header {
        font-size: 2.5rem;
        color: #FF9900;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Subtitle */
    .subtitle {
        color: white;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    /* Chat messages */
    .chat-message {
        padding: 1.5rem;
        border-radius: 1.5rem;
        margin: 1rem 0;
        max-width: 90%;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
    }
    .bot-message {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    /* Scrollbar */
    .stApp ::-webkit-scrollbar {
        width: 8px;
    }
    .stApp ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    .stApp ::-webkit-scrollbar-thumb {
        background: #FF9900;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_pipeline():
    """Load the RAG pipeline once"""
    
    # Getting HuggingFace Emeddings
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Getting Google Gemini Emeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    ) 

    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Downloading HuggingFace LLM to LOCAL - No API needed!
    # print("🔄 Loading local model...")
    # model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id)
    
    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     #max_new_tokens=150,
    #     temperature=0.4,
    #     do_sample=True,
    #     pad_token_id=tokenizer.eos_token_id
    # )
    
    # llm = HuggingFacePipeline(pipeline=pipe)
    
    # For Google Gemini LLM (Need Google API Key)
    llm = ChatGoogleGenerativeAI(api_key=os.environ.get("GOOGLE_API_KEY"),model="gemini-2.5-flash",temperature=0.5)
    
    def format_docs(docs):
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    
    qa_prompt = ChatPromptTemplate.from_template("""
    You are an AWS documentation expert.
    RULES:
    - You ONLY answer questions related to Amazon Web Services (AWS).
    - You MUST use ONLY the provided context to answer those questions which are related to AWS.
    - If a question is NOT related to AWS, politely refuse.
    - Do NOT guess or hallucinate.
    - Don't quote the context, instead synthesize a concise answer on the basis on context.
    - Don't quote any refernce or don't mention the source of the context and also never say "as per the document"/"According to the context" in your answer.
    - If the answer is not found in the context, say you don't know.

    CONTEXT: {context}

    QUESTION: {input}

    Provide a concise and factual answer:

    ANSWER:""")
    
    rag_chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: retriever.invoke(x["input"])) | format_docs
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    with st.spinner("Loading AWS RAG Model..."):
        st.session_state.rag_chain = load_rag_pipeline()

# Header
header_container = st.container()
with header_container:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">☁️ AWS Documentation Chatbot</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Powered by AWS Docs + RAG** - Ask me anything about AWS services!</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ Info")
    st.info("""
    **Features:**
    - Official AWS documentation
    - RAG-powered accurate answers
    - Chat history preserved
    - Fast local inference
    """)
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about AWS services..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke({"input": prompt})
                st.markdown(response.strip())
                st.session_state.messages.append({"role": "assistant", "content": response.strip()})
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": "Sorry, try asking about AWS services!"})

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit + LangChain + AWS Docs*")
