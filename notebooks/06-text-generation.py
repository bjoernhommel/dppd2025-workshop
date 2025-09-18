import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    get_history, set_history = mo.state([])

    def add_to_history(entry):
        set_history(lambda x: x + [entry])
    return add_to_history, get_history, set_history


@app.cell
def _(go, top_k_input):
    def plot_next_token_distribution(df):

        subset = df.head(top_k_input.stop + top_k_input.value)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=subset['token'],
            y=subset['prob'],
            marker=dict(
                color='black',
                line=dict(width=0)
            ),
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>Probability: %{y:.6f}<extra></extra>'
        ))

        fig.update_layout(
            template="none",
            showlegend=False,
            xaxis=dict(
                title=dict(text='Token', font=dict(color='black', size=12)),
                showticklabels=True,
                tickfont=dict(color='black', size=10),
                tickangle=-45,
                showgrid=False
            ),
            yaxis=dict(
                title=dict(text='Probability', font=dict(color='black', size=12)),
                showticklabels=True,
                tickfont=dict(color='black', size=10),
                showgrid=False
            ),
            margin=dict(l=50, r=20, t=20, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    return (plot_next_token_distribution,)


@app.cell
def _(mo):
    mo.md(
        """
    # Basic Text Generation
    In this exercise, we'll explore automatic item generation (AIG) as a simple use case for basic text generation. Specifically, we'll use the [bandura-v1](https://huggingface.co/magnolia-psychometrics/bandura-v1) model which is a version of a smaller gpt-2 model, fine-tuned for AIG ([Hommel et al., 2022](https://link.springer.com/article/10.1007/s11336-021-09823-9)).

    ## Background
    In this work, we trained gpt-2 on construct-item pairs obtained from the [International Personality Item Pool](https://ipip.ori.org/AlphabeticalItemList.htm), using a simple encoding template. Specifically, construct labels that applied to an item were concatenated by a delimiter (i.e., "#") and this concatenated string of constructs was in turn concatenated with the item text, using a different delimiter (i.e., "@"). For example:

        > `#extraversion#sociability@I am the life of the party`

    The idea was that the decoder model would learn the association between construct labels and thereby enable item generation for targeted constructs, by prompting in the following way:

    > `#pessimism@`

    Remember that in-context learning (e.g., Instructional prompting) wasn't possible until gpt-3.5, which was released shortly after this paper was published.

    Let us now use this model to illustrate some basic concepts of language modeling, such as sampling strategies. This marimo notebook is a bit more complicated than the other exercises, so let us disregard most of the underlying code and focus on the mechanics of text generation instead. We therefore run this notebook in app-mode.
    """
    )
    return


@app.cell
def _():
    import torch
    import pandas as pd
    import plotly.graph_objects as go
    from transformers import AutoTokenizer, GPT2LMHeadModel
    return AutoTokenizer, GPT2LMHeadModel, go, pd, torch


@app.cell
def _(AutoTokenizer, GPT2LMHeadModel, mo):
    model_path = "magnolia-psychometrics/bandura-v1"

    with mo.status.spinner("Loading model..."):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
    return model, tokenizer


@app.cell
def _(mo):
    sampling_strategy_input = mo.ui.radio(
        options=["Greedy Search", "Beam Search", "Multinomial Sampling"],
        value="Greedy Search"
    )
    sampling_strategy_intro = """
    ## Sampling Strategy

    As you may recall from the previous slides, we discussed three distinct sampling strategies, which we can use to manipulate the next-token probabilities.
    """
    mo.vstack([
        mo.md(sampling_strategy_intro),
        sampling_strategy_input    
    ])
    return (sampling_strategy_input,)


@app.cell
def _(mo, sampling_strategy_input):
    generation_intro = """
    We can add one or multiple constructs for which we want to generate items. We can also set a prefix, which will force the model to generate an item with these predefined tokens. Beam search will generate as many items as defined by the number of beams to return, which must be equal to or less than the number of beams used overall.
    """

    prefix_input = mo.ui.text(
        label="Prefix", 
        full_width=True
    )

    construct_input = mo.ui.text_area(
        label="Construct(s)",
        value="Pessimism",
        full_width=True
    )

    num_beams_input = mo.ui.slider(
        label="Number of beams", 
        start=1, 
        stop=5,
        full_width=True
    )

    num_return_sequences_input = mo.ui.slider(
        label="Number of beams to return", 
        start=1, 
        stop=5,
        full_width=True
    )

    temperature_input = mo.ui.slider(
        label="Temperature", 
        start=0.1, 
        stop=10,
        step=.1,
        value=1,
        full_width=True
    )

    top_k_input = mo.ui.slider(
        label="Top k (0 to disable)",
        start=0, 
        stop=100,
        value=0,
        full_width=True
    )

    top_p_input = mo.ui.slider(
        label="Top p (1 to disable)",
        start=.01, 
        stop=1.00,
        step=.01,
        value=1.00,
        full_width=True
    )

    run_button = mo.ui.run_button(
        label="Generate Item(s)",
        full_width=True
    )
    clear_button = mo.ui.run_button(
        label="Clear Generated Item(s)",
        full_width=True

    )

    if sampling_strategy_input.value == "Greedy Search":
        second_column = mo.vstack([])
    elif sampling_strategy_input.value == "Beam Search":
        second_column = mo.vstack([num_beams_input, num_return_sequences_input])
    elif sampling_strategy_input.value == "Multinomial Sampling":
        second_column = mo.vstack([temperature_input, top_k_input, top_p_input])

    mo.vstack([
        mo.md(generation_intro),
        mo.hstack([
            mo.vstack([            
                construct_input,
                prefix_input,
            ]),
            second_column,    
        ]),
        run_button,
        clear_button
    ])
    return (
        clear_button,
        construct_input,
        num_beams_input,
        num_return_sequences_input,
        prefix_input,
        run_button,
        temperature_input,
        top_k_input,
        top_p_input,
    )


@app.cell
def _(construct_input, mo, prefix_input, tokenizer):
    construct_sep = '#'
    item_sep = '@'

    construct_input_values = construct_input.value.split("\n")
    constructs = [x.strip() for x in construct_input_values if len(x) > 0]

    encoded_constructs = construct_sep + construct_sep.join([x.lower() for x in constructs])
    encoded_prompt = f'{encoded_constructs}{item_sep}{prefix_input.value}'

    input_tokens = tokenizer(encoded_prompt, return_tensors="pt")

    encoding_intro = f"""
    Our settings from above will pass the following prompt to the model: 

    > `{encoded_prompt}`

    Using this input, the decoder will then attempt to generate an item text, token-by-token.
    """
    mo.md(encoding_intro)
    return constructs, input_tokens


@app.cell
def _(mo):
    mo.md(r"""The tokenized encoding schema from above, as well as the sampling strategy settings, get passed to the following function:""")
    return


@app.cell
def _(
    input_tokens,
    mo,
    model,
    num_beams_input,
    num_return_sequences_input,
    sampling_strategy_input,
    temperature_input,
    tokenizer,
    top_k_input,
    top_p_input,
):
    def generate():
        outputs = model.generate(
            inputs=input_tokens.input_ids,
            attention_mask=input_tokens.attention_mask,
            max_new_tokens=50,
            num_return_sequences=num_return_sequences_input.value,
            num_beams=num_beams_input.value,
            do_sample=sampling_strategy_input.value == "Multinomial Sampling",
            temperature=temperature_input.value,
            top_k=top_k_input.value,
            top_p=top_p_input.value,
            pad_token_id=tokenizer.eos_token_id
        )
        return outputs

    mo.show_code()
    return (generate,)


@app.cell
def _(
    add_to_history,
    clear_button,
    constructs,
    generate,
    input_tokens,
    mo,
    num_beams_input,
    num_return_sequences_input,
    prefix_input,
    run_button,
    set_history,
    torch,
):
    if run_button.value:
        mo.stop(
            predicate=num_return_sequences_input.value > num_beams_input.value, 
            output="Error: The number of beams to return cannot exceed the number of beams!"
        )
        with mo.status.spinner("Generating items..."):
            with torch.no_grad():
                outputs = generate()
            add_to_history({
                'prefix': prefix_input.value,
                'constructs': constructs,
                'inputs': input_tokens.input_ids,
                'outputs': outputs,
            })

    if clear_button.value:
        set_history([])
    return


@app.cell
def _(get_history, mo, pd, tokenizer, torch):
    mo.stop(len(get_history()) < 1, "Generate items to continue...")

    def parse_output():
        data = []
        for record in get_history():
            input_tokens = record["inputs"][0]
            start_pos = len(input_tokens)

            for output_tokens in record["outputs"]:
                eos_pos = torch.where(output_tokens == tokenizer.eos_token_id)[0][0].item()
                item_tokens = output_tokens[start_pos:eos_pos]

                data.append({
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,                
                    'constructs': record["constructs"],
                    'prefix': "<NONE>" if len(record["prefix"]) == 0 else record["prefix"],
                    'item': tokenizer.decode(item_tokens),
                })
        return data

    parsed_output = parse_output()

    item_table_intro = """
    In this table you'll find the texts which have been generated by the model, hopefully resembling useful questionnaire items. Use the checkbox to select an item to explore the prediction step-by-step.
    """

    item_table = mo.ui.table(
        data=pd.DataFrame(parsed_output).drop(labels=['input_tokens', 'output_tokens'], axis=1, inplace=False), 
        selection="single"
    )

    mo.vstack([
        mo.md(item_table_intro),
        item_table
    ])
    return item_table, parsed_output


@app.cell
def _(item_table, mo, parsed_output):
    prediction_step_intro = """
    The slider shows the position of the prediction step, after the prompt. You can manipulate the slider to step back or forth in the prediction sequence, to examine the distribution at that specific point in time.
    """

    if not item_table.value.empty:

        index = item_table.value.index.item()
        output_tokens = parsed_output[index]['output_tokens']
        item_pos = len(parsed_output[index]['input_tokens'])

        prediction_step_input = mo.ui.slider(
            label="Prediction step",
            start=1,
            value=item_pos,
            stop=len(output_tokens),
            full_width=True
        )

        mo.output.append(
            mo.vstack([
                mo.md(prediction_step_intro),
                prediction_step_input
            ])        
        )
    return output_tokens, prediction_step_input


@app.cell
def _(item_table, mo, model, output_tokens, prediction_step_input, torch):
    if not item_table.value.empty:
        with mo.status.spinner():
            tokens_ids_at_step = output_tokens[0:prediction_step_input.value]        

            with torch.no_grad():
                logits = model(output_tokens.unsqueeze(0)).logits
                # for the next token
                next_token_probs = torch.softmax(logits[0, prediction_step_input.value-1, :], dim=-1)

            # for "past" tokens
            past_tokens = tokens_ids_at_step[1:]
            past_tokens_pos = torch.arange(len(past_tokens))

            past_logits = logits[0, past_tokens_pos, past_tokens]
            probs_per_step = torch.softmax(logits[0, past_tokens_pos], dim=-1)
            past_probs = probs_per_step[torch.arange(len(past_tokens)), past_tokens]
            past_log_probs = torch.log(past_probs.clamp_min(1e-12))
    return next_token_probs, past_log_probs, past_probs, tokens_ids_at_step


@app.cell
def _(
    item_table,
    mo,
    past_log_probs,
    past_probs,
    prediction_step_input,
    tokenizer,
    tokens_ids_at_step,
    torch,
):
    def render_ppl_equation(probs, ppl):
        prob_values_list = [f"{x:.3f}" for x in probs.tolist()]
        n_tokens = len(probs)

        if n_tokens >= 5:
            # Show first 2, middle placeholder, last 2
            log_terms = [
                f"\\log({prob_values_list[0]})",
                f"\\log({prob_values_list[1]})",
                "[...]",
                f"\\log({prob_values_list[-2]})",
                f"\\log({prob_values_list[-1]})"
            ]
        else:
            # Show all terms
            log_terms = [f"\\log({prob})" for prob in prob_values_list]

        return rf'''                
        \[
        \text{{ppl}} = \exp\left(-\frac{{1}}{{{n_tokens}}} \left[{" + ".join(log_terms)}\right]\right) = {ppl.item():.4f}
        \]
        '''

    prediction_step_summary_intro = """
    Here, you'll find a summary for the current prediction step. This is a good occasion to demonstrate the calculation of perplexity, which as previously discussed, is a measure of how much the predicted probability deviates from the observed probability of the generated sequence.
    """

    if not item_table.value.empty:
        tokens_at_step = [tokenizer.decode(x) for x in tokens_ids_at_step]
        text_at_step = tokenizer.decode(tokens_ids_at_step)
        display_past_probs = torch.cat([torch.tensor([float('inf')]), past_probs])
        perplexity = torch.exp(-past_log_probs.mean())
        perplexity_equation = render_ppl_equation(past_probs, perplexity)


        mo.output.append(
            mo.vstack([
                mo.md(prediction_step_summary_intro),
                mo.md(f"### Prediction step #{prediction_step_input.value}:"),
                mo.md(f'### History: "{text_at_step}"'),
                mo.md(f"### Token Ids: {str(tokens_ids_at_step.tolist())}"),
                mo.md(f'### Tokens: {str(tokens_at_step)}'),
                mo.md(f"### Probabilities: {[f'{x:.3f}' if torch.isfinite(x) else 'inf' for x in display_past_probs]}"),
                mo.md(f"### Perplexity (ppl): {perplexity.item():.3f}"),
                mo.md(perplexity_equation)
            ])
        )
    return


@app.cell
def _(item_table, mo):
    if not item_table.value.empty:
        mo.md("""
        Finally, let's have a look at the next-token probability distribution at this particular prediction step, as shown by the plot and the table below. Try changing the `top_k`, `top_p`, and `temperature` parameters above, to see how they influence the distribution.
        """)
    return


@app.cell
def _(
    item_table,
    next_token_probs,
    pd,
    temperature_input,
    tokenizer,
    top_k_input,
    top_p_input,
    torch,
):
    def create_next_token_df(next_token_probs, tokenizer, temperature, top_k, top_p):

        logits = torch.log(next_token_probs + 1e-9) / temperature
        probs = torch.softmax(logits, dim=-1)

        df = pd.DataFrame({
            'token_id': range(len(probs)),
            'token': [tokenizer.decode(i) for i in range(len(probs))],
            'prob': probs.detach().numpy()
        }).sort_values(by="prob", ascending=False).reset_index(drop=True)

        if top_k > 0:
            df.loc[top_k:, 'prob'] = 0

        if top_p < 1.0:
            cutoff = (df['prob'].cumsum() <= top_p).sum()
            cutoff = max(1, cutoff)
            df.loc[cutoff:, 'prob'] = 0

        df['prob'] /= df['prob'].sum()

        return df

    if not item_table.value.empty:

        next_token_df = create_next_token_df(
            next_token_probs=next_token_probs,
            tokenizer=tokenizer, 
            temperature=temperature_input.value,
            top_k=top_k_input.value,
            top_p=top_p_input.value,
        )
    return (next_token_df,)


@app.cell
def _(item_table, mo, next_token_df, plot_next_token_distribution):
    if not item_table.value.empty:
        plot = plot_next_token_distribution(next_token_df)

        mo.output.append(
            mo.ui.plotly(plot)
        )
    return


@app.cell
def _(item_table, mo, next_token_df):
    if not item_table.value.empty:
        mo.output.append(
            mo.ui.table(next_token_df)
        )
    return


if __name__ == "__main__":
    app.run()
