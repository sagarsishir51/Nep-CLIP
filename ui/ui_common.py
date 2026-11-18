# ui_common.py
"""Shared UI pieces used by both task screens."""
import gradio as gr
import core


def model_selector():
    """Model dropdown + Load button + status line. Returns the components."""
    with gr.Row():
        dd = gr.Dropdown(choices=core.MODEL_NAMES, value=core.MODEL_NAMES[0], label="Model", scale=4)
        load = gr.Button("Load", variant="primary", scale=1)
    status = gr.Markdown(f"Model '{core.MODEL_NAMES[0]}' selected — click **Load** to use it.")
    return dd, load, status


def start_loading(name):
    return (
        gr.update(value="Loading…", interactive=False),
        f"⏳ Loading model '{name}'…",
        gr.update(interactive=False),
    )

def load_selected_model(name):
    core.get_model(name)
    return (
        gr.update(value="Load", interactive=True),
        f"✅ Model '{name}' loaded.",
        gr.update(interactive=True),
    )

def on_model_change(name):
    return (
        f"Model '{name}' selected — click **Load** to use it.",
        gr.update(interactive=False),
    )


def wire_model_selector(dd, load, status, action_btn):
    """Attach the standard Load / dropdown-change behavior.
    `action_btn` is the screen's main button (Search or Classify) to enable/disable."""
    load.click(start_loading, [dd], [load, status, action_btn]) \
        .then(load_selected_model, [dd], [load, status, action_btn])
    dd.change(on_model_change, [dd], [status, action_btn])

# --- add to ui_common.py ---

# ui_common.py
def dual_model_selector():
    with gr.Row(equal_height=True):
        dd_a = gr.Dropdown(choices=core.MODEL_NAMES, value=core.MODEL_NAMES[0],
                           label="Model A", scale=5, min_width=180)
        dd_b = gr.Dropdown(choices=core.MODEL_NAMES,
                           value=core.MODEL_NAMES[min(1, len(core.MODEL_NAMES) - 1)],
                           label="Model B", scale=5, min_width=180)
        load = gr.Button("Load both", variant="primary", scale=1, min_width=120)
    return dd_a, dd_b, load   # status no longer created here

def start_loading_both(name_a, name_b):
    return (
        gr.update(value="Loading…", interactive=False),
        f"⏳ Loading '{name_a}' and '{name_b}'…",
        gr.update(interactive=False),
    )

def load_both_models(name_a, name_b):
    core.get_model(name_a)
    core.get_model(name_b)
    return (
        gr.update(value="Load both", interactive=True),
        f"✅ Loaded '{name_a}' and '{name_b}'.",
        gr.update(interactive=True),
    )

def on_dual_change(name_a, name_b):
    return (
        f"Models '{name_a}' & '{name_b}' selected — click **Load both**.",
        gr.update(interactive=False),
    )


def wire_dual_selector(dd_a, dd_b, load, status, action_btn):
    load.click(start_loading_both, [dd_a, dd_b], [load, status, action_btn]) \
        .then(load_both_models, [dd_a, dd_b], [load, status, action_btn])
    dd_a.change(on_dual_change, [dd_a, dd_b], [status, action_btn])
    dd_b.change(on_dual_change, [dd_a, dd_b], [status, action_btn])

def xai_compare_panel(intro):
    """Collapsible XAI block for the compare screens. Returns a dict of components."""
    with gr.Accordion("🧠 XAI analysis", open=True):
        with gr.Row(equal_height=True):
            info = gr.Markdown(intro)
            btn = gr.Button("Analyze with XAI", variant="primary", scale=0, min_width=180)
        with gr.Row():
            method = gr.Dropdown(choices=core.XAI_METHODS, value=core.XAI_METHODS[0],
                                 label="Method", scale=2)
            style = gr.Dropdown(choices=["heatmap", "box", "contour"],
                                value="heatmap", label="Overlay", scale=2)
        with gr.Row():
            with gr.Column():
                status_a = gr.Textbox(label="XAI — Model A", interactive=False)
                panel_a = gr.Image(label="Grounding A", type="filepath")
            with gr.Column():
                status_b = gr.Textbox(label="XAI — Model B", interactive=False)
                panel_b = gr.Image(label="Grounding B", type="filepath")
    return dict(info=info, method=method, style=style, btn=btn,
                status_a=status_a, status_b=status_b, panel_a=panel_a, panel_b=panel_b)
