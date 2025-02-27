import streamlit as st # UI 
import autogen
import tempfile
from autogen import ConversableAgent # autogen for AI frameawork
from autogen.coding import LocalCommandLineCodeExecutor #executor library

# Configuration for the agents
config_list_mistral = [
    {
        'model': "deepseek-r1:latest", #model name
        "base_url": "http://localhost:11434/v1", #ollama url
        "api_key": "ollama", #api key anything
    }
]


llm_mistral_config = {
    'config_list': config_list_mistral #llm defining
}
llm_codellama_config = {
    'config_list': config_list_mistral
}

# The code writer agent's system message
code_writer_system_message = """You are a helpful AI assistant.
Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
Reply 'TERMINATE' in the end when everything is done.
"""

temp_dir = tempfile.TemporaryDirectory()

# Create a local command line code executor.
executor = LocalCommandLineCodeExecutor(
    timeout=10,  # Timeout for each code execution in seconds.
    work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
)

# Create an agent with code executor configuration.
code_executor_agent = ConversableAgent(
    "code_executor_agent",
    llm_config=False,  # Turn off LLM for this agent.
    code_execution_config={"executor": executor},  # Use the local command line code executor.
    human_input_mode="ALWAYS",  # Always take human input for this agent for safety.
)

code_writer_agent = ConversableAgent(
    "code_writer_agent",
    system_message=code_writer_system_message,
    llm_config=llm_codellama_config,  # Turn off LLM for this agent.
    code_execution_config=False,  # Turn on the code executor for this agent.
)

student_agent = ConversableAgent(
    name="Student_Agent",
    system_message="You are a student willing to learn.",
    llm_config=llm_mistral_config,
    is_termination_msg=lambda msg: "good bye" in msg["content"].lower() # agent to stop
    
)
teacher_agent = ConversableAgent(
    name="Teacher_Agent",
    system_message="You are a math teacher.",
    llm_config=llm_mistral_config,
)

# Streamlit UI
st.title("AI Task Selector")
st.markdown("<h2 style='color: blue;'>Choose your task</h2>", unsafe_allow_html=True)

task = st.selectbox("Select the task you want to do:", ["Select", "Coding", "Theoretical"])

user_message = st.text_input("Enter your message:")

if task == "Theoretical" and user_message:
    if st.button("Start Theoretical Task"):
        chat_result = student_agent.initiate_chat(
            teacher_agent,
            message=user_message + " and in the End of answers say GOOD BYE",
            summary_method="reflection_with_llm",
            max_turns=1, #number of conversation
        )
        st.write("Agent's response:")
        st.write(chat_result.summary)
elif task == "Coding" and user_message:
    if st.button("Start Coding Task"):
        chat_result = code_executor_agent.initiate_chat(
            code_writer_agent,
            message=user_message + " and in the End of answers say GOOD BYE",
            max_turns=1,
        )
        st.write("Agent's response:")
        st.write(chat_result.summary)
    