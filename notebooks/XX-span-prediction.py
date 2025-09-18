import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import outlines
    from io import BytesIO
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from pydantic import BaseModel
    from typing import Literal
    return AutoModelForCausalLM, AutoTokenizer, BaseModel, BytesIO, pd


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, mo):
    model_name = "Qwen/Qwen3-8B"

    with mo.status.spinner():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto"
        )
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


@app.cell
def _(mo):
    file_upload = mo.ui.file(
        label="Upload File",
        filetypes=[".csv", ".parquet"],
        multiple=False
    )

    mo.vstack([
        file_upload
    ])
    return (file_upload,)


@app.cell
def _(BytesIO, file_upload, pd):
    if len(file_upload.value) > 0:

        file_bytes = file_upload.value[0].contents

        if file_upload.value[0].name.endswith(".parquet"):
            input_df = pd.read_parquet(BytesIO(file_bytes))
        else:
            input_df = pd.read_csv(BytesIO(file_bytes))
    return (input_df,)


@app.cell
def _(file_upload, input_df, mo):
    if len(file_upload.value) > 0:
        text_column_input  = mo.ui.dropdown(
            allow_select_none=False,
            full_width=True,
            label="Select column for corrupted text",
            value=input_df.columns.tolist()[0],
            options=input_df.columns.tolist()
        )

        target_column_input  = mo.ui.dropdown(
            allow_select_none=False,
            full_width=True,
            label="Select column for tokens to predict",
            value=input_df.columns.tolist()[0],
            options=input_df.columns.tolist()
        )


        difficulty_column_input  = mo.ui.dropdown(
            allow_select_none=True,
            full_width=True,
            label="Select column for item difficulty",
            options=input_df.columns.tolist()
        )

        mo.output.append(
            mo.vstack([
                mo.ui.table(input_df, page_size=5),
                text_column_input,
                target_column_input,
                difficulty_column_input
            ])
        )
    return


@app.function
def prompt_span_prediction():
    pass


@app.cell
def _(mo):
    max_new_tokens_input = mo.ui.number(
        label="Max New Tokens",
        start=1,
        stop=32768,
        value=1000
    )

    enable_thinking_input = mo.ui.radio(
        label="Enable Thinking", 
        options=["True", "False"], 
        value="True"
    )

    run_button = mo.ui.run_button(
        label="click"
    )

    mo.vstack([
        max_new_tokens_input,
        enable_thinking_input,
        run_button
    ])
    return enable_thinking_input, run_button


@app.cell
def _():
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    return (messages,)


@app.cell
def _(
    BaseModel,
    enable_thinking_input,
    messages,
    mo,
    model,
    run_button,
    tokenizer,
):
    if run_button.value:

        class TargetWord(BaseModel):
            target_word: str

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=bool(enable_thinking_input.value)
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            max_new_tokens=1000
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        if bool(enable_thinking_input.value):
            think_index = len(output_ids) - output_ids[::-1].index(tokenizer.vocab["</think>"])
        else:
            think_index = 0

        thinking_content = tokenizer.decode(output_ids[:think_index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[think_index:], skip_special_tokens=True).strip("\n")

        mo.output.append(
            mo.vstack([
                mo.md(thinking_content),
                mo.md(content),
            ])
        )
    return


if __name__ == "__main__":
    app.run()
