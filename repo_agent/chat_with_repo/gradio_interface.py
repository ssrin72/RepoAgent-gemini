import os
import gradio as gr
import markdown
from pathlib import Path

from repo_agent.log import logger
from repo_agent.settings import SettingsManager
from repo_agent.chat_with_repo.rag import RepoAssistant


class GradioInterface:
    def __init__(self, respond_function):
        self.respond = respond_function
        self.cssa = """
                <style>
                        .outer-box {
                            border: 1px solid #333; /* 外框的边框颜色和大小 */
                            border-radius: 10px; /* 外框的边框圆角效果 */
                            padding: 10px; /* 外框的内边距 */
                        }

                        .title {
                            margin-bottom: 10px; /* 标题和内框之间的距离 */
                        }

                        .inner-box {
                            border: 1px solid #555; /* 内框的边框颜色和大小 */
                            border-radius: 5px; /* 内框的边框圆角效果 */
                            padding: 10px; /* 内框的内边距 */
                        }

                        .content {
                            white-space: pre-wrap; /* 保留空白符和换行符 */
                            font-size: 16px; /* 内容文字大小 */
                            height: 405px;
                            overflow: auto;
                        }
                    </style>
                    <div class="outer-box"">
        
        """
        self.cssb = """
                        </div>
                    </div>
                </div>
        """
        self.setup_gradio_interface()

    def wrapper_respond(self, msg_input, system_input):
        # 调用原来的 respond 函数
        msg, output1, output2, output3, code, codex = self.respond(
            msg_input, system_input
        )
        output1 = markdown.markdown(str(output1))
        output2 = markdown.markdown(str(output2))
        code = markdown.markdown(str(code))
        output1 = (
            self.cssa
            + """
                          <div class="title">Response</div>
                            <div class="inner-box">
                                <div class="content">
                """
            + str(output1)
            + """
                        </div>
                    </div>
                </div>
                """
        )
        output2 = (
            self.cssa
            + """
                          <div class="title">Embedding Recall</div>
                            <div class="inner-box">
                                <div class="content">
                """
            + str(output2)
            + self.cssb
        )
        code = (
            self.cssa
            + """
                          <div class="title">Code</div>
                            <div class="inner-box">
                                <div class="content">
                """
            + str(code)
            + self.cssb
        )

        return msg, output1, output2, output3, code, codex

    def clean(self):
        msg = ""
        output1 = gr.HTML(
            self.cssa
            + """
                                        <div class="title">Response</div>
                                            <div class="inner-box">
                                                <div class="content">
                      
                                            """
            + self.cssb
        )
        output2 = gr.HTML(
            self.cssa
            + """
                                        <div class="title">Embedding Recall</div>
                                            <div class="inner-box">
                                                <div class="content">
                                    
                                            """
            + self.cssb
        )
        output3 = ""
        code = gr.HTML(
            self.cssa
            + """
                                        <div class="title">Code</div>
                                            <div class="inner-box">
                                                <div class="content">
                                   
                                            """
            + self.cssb
        )
        codex = ""
        return msg, output1, output2, output3, code, codex

    def setup_gradio_interface(self):
        with gr.Blocks() as demo:
            gr.Markdown("""
                # RepoAgent: Chat with doc
            """)
            with gr.Tab("main chat"):
                with gr.Row():
                    with gr.Column():
                        msg = gr.Textbox(label="Question Input", lines=4)
                        system = gr.Textbox(
                            label="(Optional)insturction editing", lines=4
                        )
                        btn = gr.Button("Submit")
                        btnc = gr.ClearButton()
                        btnr = gr.Button("record")

                    output1 = gr.HTML(
                        self.cssa
                        + """
                                        <div class="title">Response</div>
                                            <div class="inner-box">
                                                <div class="content">
                      
                                            """
                        + self.cssb
                    )
                with gr.Row():
                    with gr.Column():
                        # output2 = gr.Textbox(label = "Embedding recall")
                        output2 = gr.HTML(
                            self.cssa
                            + """
                                        <div class="title">Embedding Recall</div>
                                            <div class="inner-box">
                                                <div class="content">
                                    
                                            """
                            + self.cssb
                        )
                    code = gr.HTML(
                        self.cssa
                        + """
                                        <div class="title">Code</div>
                                            <div class="inner-box">
                                                <div class="content">
                                   
                                            """
                        + self.cssb
                    )
                    with gr.Row():
                        with gr.Column():
                            output3 = gr.Textbox(label="key words", lines=2)
                            output4 = gr.Textbox(label="key words code", lines=14)

            btn.click(
                self.wrapper_respond,
                inputs=[msg, system],
                outputs=[msg, output1, output2, output3, code, output4],
            )
            btnc.click(
                self.clean, outputs=[msg, output1, output2, output3, code, output4]
            )
            msg.submit(
                self.wrapper_respond,
                inputs=[msg, system],
                outputs=[msg, output1, output2, output3, code, output4],
            )  # Press enter to submit

        gr.close_all()
        demo.queue().launch(share=False, height=800)


# 使用方法
if __name__ == "__main__":
    # --- Configuration for testing ---
    # Set your Gemini API key as an environment variable or replace os.getenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. Please set it to your Gemini API key."
        )

    # Path to your sample repository (e.g., current directory or a specific test repo)
    # For a real test, you should point this to the root of a Python project.
    target_repo_path = Path("./")  

    # Define a path for the database file
    db_path = "./repo_docs_db.json"

    # Initialize settings manager
    SettingsManager.initialize_with_params(
        target_repo=target_repo_path,
        markdown_docs_name="markdown_docs",
        hierarchy_name=".project_doc_record",
        ignore_list=[],
        language="English",
        max_thread_count=4,
        log_level="INFO",
        weak_model_name="gemini-1.5-flash",  # Or your preferred weak model
        strong_model_name="gemini-1.5-pro",  # Or your preferred strong model
        temperature=0.2,
        request_timeout=60,
        gemini_api_key=gemini_api_key,
    )

    # Initialize RepoAssistant
    repo_assistant = RepoAssistant(db_path=db_path)

    def respond_function(msg_input, system_input):
        # Call the respond method of the initialized RepoAssistant instance
        return repo_assistant.respond(msg_input, system_input)

    gradio_interface = GradioInterface(respond_function)
