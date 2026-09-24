import gradio as gr

def add_numbers(Num1, Num2):
    return Num1 + Num2

# Define the interface
demo = gr.Interface(
    fn=add_numbers,
    inputs=[
        gr.Number(label="Number 1"),
        gr.Number(label="Number 2")
    ],
    outputs=gr.Number(label="Result")
)

# Launch the interface


def sentence_builder(
    quantity,
    tech_worker_type,
    countries,
    place,
    activity_list,
    morning
):
    countries = countries or []
    activity_list = activity_list or []

    return (
        f'The {quantity} {tech_worker_type}s from '
        f'{" and ".join(countries)} went to the {place} where they '
        f'{" and ".join(activity_list)} until the '
        f'{"morning" if morning else "night"}'
    )

demo = gr.Interface(
    fn=sentence_builder,
    inputs=[
        gr.Slider(
            3, 20,
            value=4,
            step=1,
            label="Count"
        ),

        gr.Dropdown(
            ["Data Scientist", "Software Developer", "Software Engineer"],
            label="Tech Worker Type"
        ),

        gr.CheckboxGroup(
            ["Canada", "Japan", "France"],
            label="Countries"
        ),

        gr.Radio(
            ["office", "restaurant", "meeting room"],
            label="Location"
        ),

        gr.Dropdown(
            ["partied", "brainstormed", "coded", "fixed bugs"],
            value=["brainstormed", "fixed bugs"],
            multiselect=True,
            label="Activities"
        ),

        gr.Checkbox(
            label="Morning"
        )
    ],

    outputs=gr.Textbox(label="Generated Sentence")
)

demo.launch(
    server_name="127.0.0.1",
    server_port=7860
)
