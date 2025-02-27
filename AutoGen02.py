import streamlit as st
import autogen
import tempfile
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor

# Configuration for AI models
models = {
    "DeepSeek": {
        "model": "deepseek-r1:latest",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
    "Mistral": {
        "model": "mistral-7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
    "Llama": {
        "model": "llama-2-7b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
    },
}

# Streamlit UI
st.title("AI Model Task Selector")
st.markdown("<h2 style='color: blue;'>Choose Your Task and Model</h2>", unsafe_allow_html=True)

selected_model = st.selectbox("Select AI Model:", list(models.keys()))
task = st.selectbox("Select the task you want to perform:", ["Select", "Text Generation", "Summarization", "Code Execution"])
user_message = st.text_area("Enter your message:")

# Set LLM config for the selected model
llm_config = {"config_list": [models[selected_model]]}

# Define AI agents
text_agent = ConversableAgent(
    "text_agent",
    system_message="You are an AI assistant that provides detailed responses to user queries.",
    llm_config=llm_config,
)

summarization_agent = ConversableAgent(
    "summarization_agent",
    system_message="You are an AI that summarizes text concisely.",
    llm_config=llm_config,
)

temp_dir = tempfile.TemporaryDirectory()
executor = LocalCommandLineCodeExecutor(timeout=10, work_dir=temp_dir.name)

code_agent = ConversableAgent(
    "code_agent",
    system_message="You are an AI that executes code and returns results.",
    llm_config=False,
    code_execution_config={"executor": executor},
    human_input_mode="ALWAYS",
)

code_writer_agent = ConversableAgent(
    "code_writer_agent",
    system_message="You are an AI that generates correct and executable code.",
    llm_config=llm_config,
)

# Execute based on task selection
if task == "Text Generation" and user_message:
    if st.button("Generate Text"):
        response = text_agent.initiate_chat(
            text_agent, message=user_message, max_turns=1
        )
        st.write("Generated Response:")
        st.write(response.summary)

elif task == "Summarization" and user_message:
    if st.button("Summarize Text"):
        response = summarization_agent.initiate_chat(
            summarization_agent, message=f"Summarize this: {user_message}", max_turns=1
        )
        st.write("Summary:")
        st.write(response.summary)

elif task == "Code Execution" and user_message:
    if st.button("Execute Code"):
        generated_code = code_writer_agent.initiate_chat(
            code_writer_agent, message=f"Write Python code for: {user_message}", max_turns=1
        )
        st.write("Generated Code:")
        st.code(generated_code.summary, language='python')
        
        execution_response = code_agent.initiate_chat(
            code_agent, message=generated_code.summary, max_turns=1
        )
        st.write("Execution Result:")
        st.write(execution_response.summary)
