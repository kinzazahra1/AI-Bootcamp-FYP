# Importing necessary libraries
import gradio as gr
from langchain_huggingface import HuggingFaceEndpoint

# Define the HuggingFaceEndpoint with the model you are using
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7)

# Define the function to call the model
def ask_question(question):
    response = llm.invoke(question)
    return response

# Create the Gradio interface
iface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    title="Question Answering with Mistral-7B",
    description="Ask any question and get an answer from the Mistral-7B model."
)

# Launch the interface
iface.launch()