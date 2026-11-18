# ui_zeroshot.py
"""Zero-shot classification screen."""
import gradio as gr
import core
from ui_common import dual_model_selector, wire_dual_selector


def build():
    """Build the zero-shot screen. Returns (screen_column, back_button)."""
    with gr.Column(visible=False) as screen:
        with gr.Row(equal_height=True):
            with gr.Column(scale=0, min_width=100):
                back = gr.Button("← Back", size="sm")
            with gr.Column(scale=10):
                gr.Markdown("## 🏷️ Zero-shot Classification — Compare")

        dd_a, dd_b, load = dual_model_selector()
        status = gr.Markdown("Select two models and click **Load both**.")

        image = gr.Image(label="Image to classify", type="filepath",
                         height=600, width=600)
        labels = gr.Textbox(label="Candidate labels (one per line)",
                            placeholder="रोटी\nखाना\nतरकारी\nफलफूल", lines=6)
        classify_btn = gr.Button("Classify both", variant="primary", interactive=False)

        with gr.Row():
            with gr.Column():
                status_a = gr.Textbox(label="Model A", interactive=False)
                result_a = gr.Label(label="Predictions A", num_top_classes=10)
            with gr.Column():
                status_b = gr.Textbox(label="Model B", interactive=False)
                result_b = gr.Label(label="Predictions B", num_top_classes=10)

        wire_dual_selector(dd_a, dd_b, load, status, classify_btn)
        classify_btn.click(
            core.classify_compare,
            [dd_a, dd_b, image, labels],
            [result_a, result_b, status_a, status_b],
        )

    return screen, back