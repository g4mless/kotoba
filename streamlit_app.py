import streamlit as st
import together
import google as genai
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
    
    # Initialize API keys in session state
    if 'together_api' not in st.session_state:
        st.session_state.together_api = ""
    if 'gemini_api' not in st.session_state:
        st.session_state.gemini_api = ""
    if 'groq_api' not in st.session_state:
        st.session_state.groq_api = ""
    
    # API Key input based on selected provider
    if provider == "Together AI":
        st.markdown("""
        ### Together AI API Key Required
        Get your API key from [Together AI](https://www.together.ai)
        """)
        together_api = st.text_input(
            'Enter Together AI API key:', 
            value=st.session_state.together_api,
            type='password',
            help="Required for Together AI models"
        )
        if together_api != st.session_state.together_api:
            st.session_state.together_api = together_api
            
        # Together AI model selection
        model_options = {
            'Llama 3.3 70B': 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
            'DeepSeek 70B': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free'
        }
        selected_model = st.selectbox('Choose Model', list(model_options.keys()), key='selected_model')
        llm = model_options[selected_model]
        
    elif provider == "Groq":
        st.markdown("""
        ### Groq API Key Required
        Get your API key from [Groq Console](https://console.groq.com)
        """)
        groq_api = st.text_input(
            'Enter Groq API key:', 
            value=st.session_state.groq_api,
            type='password',
            help="Required for Groq models"
        )
        if groq_api != st.session_state.groq_api:
            st.session_state.groq_api = groq_api
            
        # Groq model selection
        selected_model = "llama3-70b-8192"
        llm = selected_model
        
        # Add top_p and max_length sliders for Groq
        top_p = st.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01,
                         help="Controls diversity via nucleus sampling")
        max_length = st.slider('Max Length', min_value=64, max_value=8192, value=1024, step=8,
                             help="The maximum number of tokens to generate.")
        
    else:  # Google Gemini
        st.markdown("""
        ### Google Gemini API Key Required
        Get your API key from [Google AI Studio](https://ai.google.dev/)
        """)
        gemini_api = st.text_input(
            'Enter Gemini API key:', 
            value=st.session_state.gemini_api,
            type='password',
            help="Required for Gemini models"
        )
        if gemini_api != st.session_state.gemini_api:
            st.session_state.gemini_api = gemini_api
            
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
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI assistant. Please provide your API key in the sidebar to start chatting. 👋"}]

# Function for generating responses
def generate_response(prompt_input):
    provider = st.session_state.provider
    
    if provider == "Together AI":
        if not st.session_state.together_api:
            st.error("Please provide your Together AI API key in the sidebar.")
            return None
            
        messages = [{"role": "system", "content": """You are a helpful AI assistant. Be direct and factual in your responses. Use bullet points for lists."""}]
        
        # Add chat history
        for dict_message in st.session_state.messages:
            messages.append({"role": dict_message["role"], "content": dict_message["content"]})
        
        # Add user's new message
        messages.append({"role": "user", "content": prompt_input})
        
        try:
            client = together.Together(api_key=st.session_state.together_api)
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
        if not st.session_state.groq_api:
            st.error("Please provide your Groq API key in the sidebar.")
            return None
            
        messages = [{"role": "system", "content": """You are a helpful AI assistant. Be direct and factual in your responses. Use bullet points for lists."""}]
        
        # Add chat history
        for dict_message in st.session_state.messages:
            messages.append({"role": dict_message["role"], "content": dict_message["content"]})
        
        # Add user's new message
        messages.append({"role": "user", "content": prompt_input})
        
        try:
            client = Groq(api_key=st.session_state.groq_api)
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
        if not st.session_state.gemini_api:
            st.error("Please provide your Google Gemini API key in the sidebar.")
            return None
            
        try:
            genai.configure(api_key=st.session_state.gemini_api)
            model = genai.GenerativeModel(selected_model)
            
            # Format chat history for Gemini with system prompt
            chat = model.start_chat(history=[])
            
            # Add system prompt for Gemini
            system_prompt = """You are a helpful AI assistant. Be direct and factual in your responses. Use bullet points for lists."""
            chat.send_message(system_prompt)
            
            # Add chat history excluding the initial greeting
            for message in st.session_state.messages:
                if message["role"] != "assistant" or message != st.session_state.messages[0]:
                    chat.send_message(message["content"])
            
            response = chat.send_message(
                prompt_input,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature
                ),
                stream=True
            )
            return response
            
        except Exception as e:
            st.error(f"Gemini API Error: {str(e)}")
            return None

def clear_chat_history():
    # Reset UI messages and chat history for both providers
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm your AI assistant. Please provide your API key in the sidebar to start chatting. 👋"}]
    
    # Reset provider-specific session states
    if "gemini_chat" in st.session_state:
        del st.session_state.gemini_chat
    
    # For Together AI, the messages array is already reset above
    # This ensures the next API call will start fresh
    
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
