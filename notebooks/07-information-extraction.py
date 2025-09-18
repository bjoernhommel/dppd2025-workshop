import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    get_image_descriptions, set_image_descriptions = mo.state([])
    get_image_bytes, set_image_bytes = mo.state([])
    get_structured_output, set_structured_output = mo.state(None)

    def add_image_descriptions(entry):
        set_image_descriptions(lambda x: x + entry)

    def add_image_bytes(entry):
        set_image_bytes(lambda x: x + [entry])
    return (
        add_image_bytes,
        add_image_descriptions,
        get_image_bytes,
        get_image_descriptions,
        get_structured_output,
        set_image_bytes,
        set_image_descriptions,
        set_structured_output,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Multimodal and Structured Information Extraction

    Instruction-tuned LLMs excel at diverse tasks, as evidenced by the success of proprietary models like GPT-5. Recent advances in open-source models, particularly following releases like DeepSeek-R1, have significantly narrowed the performance gap. Today's medium-sized open-source LLMs can run locally while delivering impressive capabilities for both text processing and structured data extraction tasks.

    In this exercise, we will see how two open-source LLMs can work together to extract information from images and produce structured outputs. In our case, we want to extract questionnaire items in a structured way. We'll use two models for this, running on our local [Ollama](https://ollama.com/) server.

    1. [`qwen2.5vl:7b`](https://ollama.com/library/qwen2.5vl): A multimodal, vision-language model which we'll use to transcribe pages from partially obscured research papers.
    2. [`qwen3:32b`](https://ollama.com/library/qwen3): A larger "thinking"-model which we'll use to parse the transcriptions to structured output.

    **Note**: This notebook requires you to run an Ollama server in the background.

    We start by importing packages.
    """
    )
    return


@app.cell
def _(mo):
    import json
    import base64
    import ollama
    import pandas as pd
    from pydantic import BaseModel
    from typing import List, Literal, Optional, Dict, Any

    mo.show_code()
    return BaseModel, List, Literal, base64, json, ollama, pd


@app.cell
def _(mo):
    mo.md(
        r"""
    Next, let's test the `qwen3:32b` model, to see if everything is working correctly. We do this by defining a simple function that will pass the required parameters to our ollama server, using the corresponding `ollama` python package. 

    Note that the `messages` list contains the entire conversation protocol. If you were to carry out a longer conversation with an LLM, your subsequent prompts and the LLM's responses simply get appended to this list. Here, there are three important `roles` that can be distinguished:

    - `system`: The system prompt sets the general instructions for the LLM at the beginning of the conversation.
    - `user`: User prompts contain the user's input.
    - `assistant`: The LLM's response to the previous `user` prompt.

    Note the hyperparameters (e.g., `temperature`) which we have previously discussed. For this particular model, it is recommended not to change these settings when using the model in thinking mode. Always check the model card before using a model.
    """
    )
    return


@app.cell
def _(mo, ollama):
    def test_model_single_prompt(system_prompt, prompt):
        response = ollama.chat(
            model='qwen3:32b',
            options={
                "seed":42,
                "temperature":0.6,
                "top_k":20,
                "top_p":.95
            },    
            messages=[
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    'role': 'user', 
                    'content': prompt
                }
            ]
        )

        return response

    mo.show_code()
    return (test_model_single_prompt,)


@app.cell
def _(mo):
    mo.md(r"""We'll define a simple user interface which we'll use to dynamically set the `system` and user `prompt`.""")
    return


@app.cell
def _(mo):
    testrun_system_prompt_input = mo.ui.text(
        label="System Prompt",
        value="You are a behavioral scientist who talks like a pirate.",
        full_width=True
    )

    testrun_prompt_input = mo.ui.text_area(
        label="User Prompt",
        value="Please tell me about the 5-Factor Model of Personality in one sentence.",
        full_width=True
    )

    testrun_button = mo.ui.run_button(
        label="Generate",
        full_width=True
    )

    mo.vstack([
        testrun_system_prompt_input,
        testrun_prompt_input,
        testrun_button
    ])
    return testrun_button, testrun_prompt_input, testrun_system_prompt_input


@app.cell
def _(mo):
    mo.md(
        r"""
    Let's take a look at the models' response to our prompt. As you can see, we get a bunch of meta-information, but the key-object of interest is the `message` property, which, in turn, contains the response in `content`.

    Notice the structure of `message` is exactly as the entries in `messages` from above? This makes it easy to simply append the response to longer conversations.

    Now, looking at the `content` we can see that parts of the model's response has been wrapped in `<think> ... </think>`-tags. The purpose of this is to mimic an internal monologue of the model, which helps it to produce better results (but also drastically increases the processing time).
    """
    )
    return


@app.cell
def _(
    mo,
    test_model_single_prompt,
    testrun_button,
    testrun_prompt_input,
    testrun_system_prompt_input,
):
    if testrun_button.value:
        with mo.status.spinner("Generating response..."):
            response = test_model_single_prompt(
                system_prompt=testrun_system_prompt_input.value,
                prompt=testrun_prompt_input.value
            )

        mo.output.append(
            mo.vstack([
                dict(response),
                mo.plain_text(response['message']['content'])
            ])
        )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Multimodal Information Extraction

    Now we get to the real task at hand, which is the extraction of content. First, we'll define a simple user-interface which lets us upload .JPG-images, which for example could be scanned research papers.

    Let us use this interface to upload some pages. In our example, we'll use a page from the paper introducing the [HEXACO–60](https://hexaco.org/) personality inventory from Ashton & Lee (2009).
    """
    )
    return


@app.cell
def _(mo):
    file_upload = mo.ui.file(
        filetypes=[".jpg"],
        multiple=True
    )
    file_upload
    return (file_upload,)


@app.cell
def _(add_image_bytes, file_upload):
    if file_upload.value:
        for file in file_upload.value:
            add_image_bytes(file.contents)
    return


@app.cell
def _(mo):
    mo.md(r"""This time, we define a function to prompt the multimodal vision model `qwen2.5vl:7b`. Note that our prompt template contains an additional `images` key, which is a list of images as bytes.""")
    return


@app.cell
def _(mo, ollama):
    def describe_image(image_bytes, prompt):

        response = ollama.chat(
            model="qwen2.5vl:7b",
            messages=[
                {
                    "role": "user", 
                    "content": prompt,
                    "images": [image_bytes]
                }
            ]
        )

        return response["message"]["content"]

    mo.show_code()
    return (describe_image,)


@app.cell
def _(mo):
    mo.md(r"""Below, you can see previews of the uploaded pages which will get passed to `qwen2.5vl:7b`. After prompting our model with it, scroll down to see the model output, which should be the transcribed text from each page.""")
    return


@app.cell
def _(base64, get_image_bytes, get_image_descriptions, mo):
    carousel_content = []

    if len(get_image_bytes()) > 0:

        base64_encoded = [base64.b64encode(x).decode("utf-8") for x in get_image_bytes()]
        images_url = [f"data:image/jpg;base64,{x}" for x in base64_encoded]
        images_html = [mo.image(src=x, height=666) for x in images_url]

        carousel_content = [
            mo.vstack([
                mo.md("`Image:`"),
                images_html[i],
                mo.md("`Extracted content:`"),
                mo.md(get_image_descriptions()[i] if i < len(get_image_descriptions()) else "")
            ], align="center") 
            for i in range(len(images_html))
        ]

    mo.carousel(carousel_content)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Next, we again define a user interface for prompting the `qwen2.5vl:7b`. We prompt the model to extract all visible text on each page and transcribe it to Markdown-format.

    ::lucide:lightbulb:: Remember that saying "please" and "thank you" may one day save your life, when LLMs have achieved world domination.
    """
    )
    return


@app.cell
def _():
    default_image_prompt = """
    Please extract all visible text from the image I've shared, using proper Markdown formatting for text, headers, lists, tables, and text styles. Ensure UTF-8 encoding. Respond with valid markdown only, and nothing else.
    """
    return (default_image_prompt,)


@app.cell
def _(default_image_prompt, mo):

    image_prompt = mo.ui.text_area(
        label="Image Prompt",
        value=default_image_prompt,
        rows=9,
        full_width=True
    )

    describe_image_button = mo.ui.run_button(
        label="Describe all the images",
        full_width=True
    )

    clear_image_descriptions_buttom = mo.ui.run_button(
        label="Reset",
        full_width=True
    )


    mo.vstack([image_prompt, describe_image_button, clear_image_descriptions_buttom])
    return clear_image_descriptions_buttom, describe_image_button, image_prompt


@app.cell
def _(
    add_image_descriptions,
    clear_image_descriptions_buttom,
    describe_image,
    describe_image_button,
    get_image_bytes,
    image_prompt,
    mo,
    set_image_bytes,
    set_image_descriptions,
):
    if describe_image_button.value:

        mo.stop(
            predicate=len(get_image_bytes()) < 1 or len(image_prompt.value) < 1, 
            output="::lucide:triangle-alert:: You need to upload images first and enter a prompt!"
        )
        image_descriptions = []

        with mo.status.progress_bar(total=len(get_image_bytes()), title="Describing images...") as bar:
            for current_image in get_image_bytes():
                image_description = describe_image(
                    image_bytes=current_image, 
                    prompt=image_prompt.value
                )

                image_descriptions.append(image_description)
                bar.update()
        add_image_descriptions(image_descriptions)

    if clear_image_descriptions_buttom.value:
        set_image_descriptions([])
        set_image_bytes([])
    return


@app.cell
def _(mo):
    mo.md(r"""Hopefully, this should have worked well! Multimodal vision-language models have a key advantage over other image-to-text extraction techniques, like OCR: They understand context and can infer partially obscured text. For the same reason, extracted content may also sometimes contain confabulated text.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Structured Information Extraction

    Okay, so now we got a bunch of text, transcribed from a scanned research paper. While this is nice to have, for quantitative analysis, we need structured data!

    One realistic use case is that we would like to parse any questionnaire data in the document in a structured way. To achieve this, we post-process the output from `qwen2.5vl:7b` with the higher-performing, thinking `qwen3:32b` model.

    We'll use this prompt template:
    """
    )
    return


@app.cell
def _(mo):
    default_structured_prompt = '''
    The following document is content from an academic paper, which may or may not contain survey, test or questionnaire:
    """
    {transcript}
    """
    If this document contains items from a survey, test or questionnaire set `is_survey` to `true`, else, set `is_survey` to `false`.

    In the case that `is_survey` is true,  for each item in this survey, test or questionnaire, extract the following:
    - `item_text`: The item text, statement or question presented to the survey respondent.
    - `scale_name`: The name of the scale, construct or trait, measured by the item.
    - `keying`: The keying of the item in relation to the construct, based on scoring instructions. Which may either be:
        - `positive`: If the item is positively associated with the scale, construct or trait.
        - `negative`: If the item is negatively associated with the scale, construct or trait (e.g., reverse-keyed).

    Respond with valid JSON array only, and nothing else.
    '''

    mo.show_code()
    return (default_structured_prompt,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Notice the `transcript` placeholder in the prompt template? This is where we later inject the output from the previous model.

    But how can we ensure that `qwen3:32b` will stick to a consistent format (i.e., a dictionary with consistent keys)? Here, `ollama`'s structured output functionality comes into play, as seen in this function:
    """
    )
    return


@app.cell
def _(BaseModel, List, Literal, mo, ollama):
    def structured_extraction_prompt(prompt):

        class Item(BaseModel):
            item_text: str
            construct_name: str
            keying: Literal["positive", "negative"] = "positive"

        class Document(BaseModel):
            thinking: str
            is_survey: bool
            items: List[Item]

        response = ollama.chat(
            model='qwen3:32b',
            options={
                "seed":42,
                "temperature":0.6,
                "top_k":20,
                "top_p":.95
            },
            messages=[
                {
                    'role': 'system',
                    'content': "You are a helpful behavioral scientist."
                },
                {
                    'role': 'user', 
                    'content': prompt
                }
            ],
            format=Document.model_json_schema()
        )

        return response["message"]["content"]

    mo.show_code()
    return (structured_extraction_prompt,)


@app.cell
def _(mo):
    mo.md(
        r"""
    We pre-define a structure by declaring two `pydantic` models, with `Item` nested in `Document`. The inner workings are quite simple, but smart: The output probabilities for each token that doesn't adhere to this structure will be constrained to 0, thus forcing the model to comply with the desired structure!

    This may normally cause a conflict with thinking-mode, as the model will always produce `<think>` tags first. We therefore use a sink-variable to "redirect" the thinking at the `Document`-level.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""To try this approach, we store the outputs from the `qwen2.5vl:7b` text extraction into a table. By selecting a row, we can now pass the content to our prompt-function for `qwen3:32b`.""")
    return


@app.cell
def _(get_image_descriptions, mo, pd):
    image_descriptions_df = pd.DataFrame(
        data=get_image_descriptions(), 
        columns=["description"]
    )

    image_descriptions_table = mo.ui.table(
        data=image_descriptions_df,
        selection="single"
    )

    image_descriptions_table
    return (image_descriptions_table,)


@app.cell
def _(
    default_structured_prompt,
    extract_survey_button,
    image_descriptions_table,
    mo,
    set_structured_output,
    structured_extraction_prompt,
):
    if not image_descriptions_table.value.empty:
        parsed_prompt = default_structured_prompt.format(
            transcript=image_descriptions_table.value.iloc[0]["description"]
        )
        mo.output.append(
            mo.vstack([
                mo.md("This is the interpolated prompt which we'll pass to the model:"),
                mo.plain_text(parsed_prompt)
            ])

        )
        if extract_survey_button.value:
            with mo.status.spinner("Generating structured output..."):
                structured_output = structured_extraction_prompt(parsed_prompt)
                set_structured_output(structured_output)

    elif extract_survey_button.value:
        mo.output.append(
            mo.md(
                text="::lucide:triangle-alert:: You need to select a row in the table of previously extracted documents!"
            )
        )
    return


@app.cell
def _(mo):
    extract_survey_button = mo.ui.run_button(
        label="Extract Survey Content",
        full_width=True
    )

    mo.vstack([
        mo.md("Press the button to send a prompt to our model with the input from above. Note that processing may take quite some time, with long documents and thinking-mode enabled."),
        extract_survey_button
    ])
    return (extract_survey_button,)


@app.cell
def _(mo):
    mo.md(r"""Finally, we parse the content from the response object with `json.loads()` to get:""")
    return


@app.cell
def _(get_structured_output, json, mo):
    if get_structured_output():
        mo.output.append(
            json.loads(get_structured_output())   
        )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Few-Shot Learning

    Thinking mode with structured outputs will give us useful results most of the time. However, sometimes, the outputs will contain artifacts, such as item numbering in `item_text`. We can try to prevent this by refining our initial prompt, but the longer the instructions, the more difficult it will be for the model to adhere to all of them.

    We can therefore combine our approach from above with a simple trick. Recall that Few-Shot Learning constitutes a form of "implicit learning" for models, where previous examples help the model to know what output the user is expecting.

    Take a look at this dummy-code:
    """
    )
    return


@app.cell
def _(mo, ollama):
    def few_shot_prompt(prompt):

        response = ollama.chat(
            model='qwen3:32b',
            options={
                "seed":42,
                "temperature":0.6,
                "top_k":20,
                "top_p":.95
            },
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful behavioral scientist."
                },
                {
                    "role": "user",
                    "content": "Please extract all the items from this survey: [...]"
                },
                {
                    "role": "assistant", 
                    "content": "<think>Okay, let's see [...]</think> - 'I am the life of the party.', - 'I accept challenging tasks.'"
                },
                {
                    "role": "user", 
                    "content": "Please extract all the items from this survey: [...]"
                }
            ]
        )

        return response["message"]["content"]

    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(r"""Here, we have simply added a fake conversation history, before prompting the model. Moreover, we simulate previous model output by specifying `"role": "assistant"` for content that, in reality, was authored by the user. As LLMs often reproduce previous output structures, we can expect more consistent responses using this approach.""")
    return


if __name__ == "__main__":
    app.run()
