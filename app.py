import gradio as gr
from agent import agent_respond

def agent_interface(user_question, debug_mode=True):
    return agent_respond(user_question)

custom_css = """
.gradio-container {
    max-width: 1400px !important;
    margin-left: auto;
    margin-right: auto;
}
.output-box {
    min-height: 500px !important;
    font-size: 16px !important;
}
.input-box {
    min-height: 150px !important;
    font-size: 16px !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
    gr.Markdown("# Medical transcripts QA agent")
    gr.Markdown("An agent that uses document retrieval and live web search to answer questions on medical transcripts.")
    
    with gr.Row():
        with gr.Column(scale=1):
            user_question = gr.Textbox(
                lines=4, 
                placeholder="Ask a healthcare question...", 
                elem_classes="input-box",
                label="Question"
            )
            debug_mode = gr.Checkbox(label="Debug Mode", value=True)
            submit_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")
        
        with gr.Column(scale=2):
            output = gr.Textbox(
                lines=30, 
                elem_classes="output-box",
                label="Response"
            )
    
    submit_btn.click(
        fn=agent_interface,
        inputs=[user_question, debug_mode],
        outputs=output
    )
    
    clear_btn.click(
        fn=lambda: "",
        inputs=None,
        outputs=[user_question, output]
    )

if __name__ == "__main__":
    demo.launch()