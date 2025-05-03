import os
import gradio as gr
import tempfile
import json
from agent import MedTranscriptAgent
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

if not os.getenv("ANTHROPIC_API_KEY"):
    logger.error("ANTHROPIC_API_KEY not found in environment variables or .env file")
    raise ValueError("ANTHROPIC_API_KEY is required. Please add it to your .env file.")

agent = MedTranscriptAgent(debug=True)

conversation_threads = {}

def process_message(message, conversation_id=None, pdf_file=None):
    if not conversation_id:
        import uuid
        conversation_id = str(uuid.uuid4())
        conversation_threads[conversation_id] = True
        logger.info(f"Created new conversation with ID: {conversation_id}")
    
    if pdf_file is not None:
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "uploaded.pdf")
        
        with open(temp_path, "wb") as f:
            f.write(pdf_file)
        
        pdf_id = agent.load_pdf(temp_path)
        logger.info(f"Loaded PDF '{pdf_id}' for conversation {conversation_id}")
    
    logger.info(f"Processing message for conversation {conversation_id}: {message}")
    response = agent.chat(message, thread_id=conversation_id)
    
    if hasattr(agent, "conversation_threads") and conversation_id in agent.conversation_threads:
        thread_msgs = agent.conversation_threads[conversation_id]
        logger.info(f"Thread {conversation_id} now has {len(thread_msgs)} messages")
    
    return response, conversation_id

with gr.Blocks(title="Medical Transcript Q&A System") as demo:
    gr.Markdown("# Medical Transcript Q&A System")
    gr.Markdown("Ask questions about medical procedures, treatments, or general medical information.")
    
    conversation_id = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500,
                type="messages",
                avatar_images=("👤", "🩺"),
                label="Conversation"
            )
            msg = gr.Textbox(
                label="Your Question", 
                placeholder="Ask a medical question...",
                lines=3
            )
            
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary", scale=2)
                clear_btn = gr.Button("Clear Conversation", scale=1)
        
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="Upload PDF Document (Optional)", 
                file_types=[".pdf"],
                type="binary"
            )
            debug_info = gr.Textbox(
                label="Debug Info", 
                visible=True, 
                interactive=False,
                lines=5
            )
    
    def user(message, history, conv_id, pdf):
        """Add user message to chat history"""
        logger.info(f"User message: {message}")
        logger.info(f"Current history length: {len(history) if history else 0}")
        logger.info(f"Current conversation ID: {conv_id}")
        
        if history is None:
            history = []
        history.append({"role": "user", "content": message})
        
        return "", history, conv_id, pdf
    
    def bot(history, conv_id, pdf):
        """Process user message and add bot response to chat history"""
        if not history or len(history) == 0:
            return history, conv_id, pdf, "Error: No message to process"
        
        user_message = history[-1]["content"]
        logger.info(f"Processing user message: {user_message[:50]}...")
        
        try:
            response, new_conv_id = process_message(user_message, conv_id, pdf)
            
            history.append({"role": "assistant", "content": response})
            
            thread_msg_count = 0
            if hasattr(agent, "conversation_threads") and new_conv_id in agent.conversation_threads:
                thread_msg_count = len(agent.conversation_threads[new_conv_id])
            
            debug_text = f"Conversation ID: {new_conv_id}\n"
            debug_text += f"UI Messages: {len(history)}\n"
            debug_text += f"Agent Thread Messages: {thread_msg_count}\n"
            debug_text += f"PDF Uploaded: {'Yes' if pdf else 'No'}\n"
            
            return history, new_conv_id, None, debug_text
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing message: {error_msg}")
            
            history.append({"role": "assistant", "content": f"Error: {error_msg}"})
            return history, conv_id, None, f"Error occurred: {error_msg}"
    
    def clear_conversation():
        """Clear the current conversation and start a new one"""
        logger.info("Clearing conversation history")
        
        import uuid
        new_id = str(uuid.uuid4())
        logger.info(f"Created new conversation with ID: {new_id}")
        
        return new_id, gr.update(value=None), [], None, f"Started new conversation with ID: {new_id}"
    
    msg.submit(
        user, 
        [msg, chatbot, conversation_id, pdf_input], 
        [msg, chatbot, conversation_id, pdf_input]
    ).then(
        bot, 
        [chatbot, conversation_id, pdf_input], 
        [chatbot, conversation_id, pdf_input, debug_info]
    )
    
    submit_btn.click(
        user, 
        [msg, chatbot, conversation_id, pdf_input], 
        [msg, chatbot, conversation_id, pdf_input]
    ).then(
        bot, 
        [chatbot, conversation_id, pdf_input], 
        [chatbot, conversation_id, pdf_input, debug_info]
    )
    
    clear_btn.click(
        clear_conversation, 
        None, 
        [conversation_id, msg, chatbot, pdf_input, debug_info]
    )

if __name__ == "__main__":
    logger.info("Starting Medical Transcript Q&A System...")
    logger.info(f"API Key present: {'Yes' if os.getenv('ANTHROPIC_API_KEY') else 'No'}")
    demo.launch()