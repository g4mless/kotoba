import streamlit as st
import together
from google import genai
from google.genai import types
import os
from groq import Groq

# App title
st.set_page_config(
    page_title="Kotoba",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better UI
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMarkdown p {
        font-size: 1.1rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .api-warning {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.title('💬 Kotoba')
    
    # Provider selection
    provider = st.selectbox(
        "Select Provider",
        ["Together AI", "Google Gemini", "Groq"],
        key="provider"
    )
    
    if provider == "Together AI":
        # Together AI model selection
        model_options = {
            'Llama 3.3 70B': 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
            'DeepSeek 70B': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free'
        }
        selected_model = st.selectbox('Choose Model', list(model_options.keys()), key='selected_model')
        llm = model_options[selected_model]
        
    elif provider == "Groq":
        # Groq model selection
        selected_model = "llama3-70b-8192"
        llm = selected_model
        
        # Add top_p and max_length sliders for Groq
        top_p = st.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01,
                         help="Controls diversity via nucleus sampling")
        max_length = st.slider('Max Length', min_value=64, max_value=8192, value=1024, step=8,
                             help="The maximum number of tokens to generate.")
        
    else:  # Google Gemini
        # Gemini model selection
        selected_model = "gemini-2.0-flash"
        llm = selected_model

    # Common parameters
    temperature = st.slider('Temperature', min_value=0.01, max_value=2.0, value=0.7, step=0.01, 
                          help="Higher values make the output more random, lower values make it more focused and deterministic.")
    
    if provider in ["Together AI", "Groq"]:
        if "top_p" not in locals():  # Only add if not already added
            top_p = st.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01,
                            help="Controls diversity via nucleus sampling")
            max_length = st.slider('Max Length', min_value=64, max_value=4096, value=1024, step=8,
                                help="The maximum number of tokens to generate.")

# Initialize session state for messages if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI assistant. How can I help you today? 👋"}]

# Function for generating responses
def generate_response(prompt_input):
    provider = st.session_state.provider
    
    if provider == "Together AI":
        try:
            together_api_key = st.secrets["together_api"]
            if not together_api_key:
                st.error("Together AI API key not found in secrets.toml")
                return None
                
            messages = [{"role": "system", "content": """You are a helpful AI assistant. Be direct and factual in your responses. Use bullet points for lists."""}]
            
            # Add chat history
            for dict_message in st.session_state.messages:
                messages.append({"role": dict_message["role"], "content": dict_message["content"]})
            
            # Add user's new message
            messages.append({"role": "user", "content": prompt_input})
            
            client = together.Together(api_key=together_api_key)
            stream = client.chat.completions.create(
                model=llm,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_length,
                stream=True
            )
            return stream
            
        except Exception as e:
            st.error(f"Together AI Error: {str(e)}")
            return None
            
    elif provider == "Groq":
        try:
            groq_api_key = st.secrets["groq_api"]
            if not groq_api_key:
                st.error("Groq API key not found in secrets.toml")
                return None
                
            messages = [{"role": "system", "content": """You are a helpful AI assistant. Be direct and factual in your responses. Use bullet points for lists."""}]
            
            # Add chat history
            for dict_message in st.session_state.messages:
                messages.append({"role": dict_message["role"], "content": dict_message["content"]})
            
            # Add user's new message
            messages.append({"role": "user", "content": prompt_input})
            
            client = Groq(api_key=groq_api_key)
            stream = client.chat.completions.create(
                model=llm,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_length,
                stream=True
            )
            return stream
            
        except Exception as e:
            st.error(f"Groq Error: {str(e)}")
            return None
            
    else:  # Google Gemini
        try:
            gemini_api_key = st.secrets["gemini_api"]
            if not gemini_api_key:
                st.error("Gemini API key not found in secrets.toml")
                return None
                
            client = genai.Client(api_key=gemini_api_key)
            
            # Format messages for Gemini - combine all messages into a single prompt
            system_prompt = """You are a helpful AI assistant. Be direct and factual in your responses. Use bullet points for lists.\n\n"""
            conversation = system_prompt
            
            # Add chat history excluding the initial greeting
            for message in st.session_state.messages:
                if message["role"] != "assistant" or message != st.session_state.messages[0]:
                    conversation += f"{message['role'].title()}: {message['content']}\n"
            
            # Add user's new message
            conversation += f"Assistant: "
            
            response = client.models.generate_content_stream(
                model=selected_model,
                contents=conversation,
                config=types.GenerateContentConfig(
                    temperature=temperature
                )
            )
            return response
            
        except Exception as e:
            st.error(f"Gemini API Error: {str(e)}")
            return None

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI assistant. How can I help you today? 👋"}]
    if "gemini_chat" in st.session_state:
        del st.session_state.gemini_chat
    
st.sidebar.button('Clear Chat History', on_click=clear_chat_history, type="primary")

# Create a container for the chat messages
chat_container = st.container()

# User input
if prompt := st.chat_input("Ask me anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Generate new response if last message is from user
    if st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):
            stream = generate_response(st.session_state.messages[-1]["content"])
            if stream:
                response_placeholder = st.empty()
                full_response = ""
                
                try:
                    if st.session_state.provider in ["Together AI", "Groq"]:
                        for chunk in stream:
                            if chunk.choices[0].delta.content is not None:
                                full_response += chunk.choices[0].delta.content
                                response_placeholder.markdown(full_response + "▌")
                    else:  # Google Gemini
                        for chunk in stream:
                            full_response += chunk.text
                            response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Error during streaming: {str(e)}")
