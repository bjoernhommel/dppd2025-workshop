import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Text Classification
    Encoder models can be trained for the task of classifying text into certain, pre-defined categories. This usually requires fine-tuning a pretrained encoder model on labeled data, where the labels correspond to the categories that the model will be tasked to predict.

    In this short exercise, we look at a common classification problem, namely sentiment analysis. In sentiment analysis, we want to categorize sentences based on their valence, usually in categories of `positive`, `negative` and sometimes `neutral` sentiment.

    We start by importing the python packages required.
    """
    )
    return


@app.cell
def _(mo):
    import pandas as pd
    from transformers import pipeline
    mo.show_code()
    return pd, pipeline


@app.cell
def _(mo):
    mo.md(r"""Because the task is well-defined, we can use a handy `pipeline` from the transformers library, which saves us a bit of work. We simply need to specify what task we want the model to solve, and what model to use. Let us again use a small but popular model for demonstration purposes, this time [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) which has been fine-tuned for sentiment analysis.""")
    return


@app.cell
def _(mo, pipeline):
    with mo.status.spinner():
        classifier = pipeline(
            task="text-classification", 
            model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
        )

    mo.show_code()
    return (classifier,)


@app.cell
def _(mo):
    mo.md(r"""Before we begin, let us again take a look at the model architecture. Because we use a pipeline, we now find the model (and the tokenizer) nested in the pipeline object `classifier`.""")
    return


@app.cell
def _(classifier, mo):
    print(classifier.model)
    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(r"""::lucide:message-circle-question-mark:: Remember the BERT model we examined in the previous exercise on masked language modeling? Can you find the crucial difference between the two?""")
    return


@app.cell
def _(mo):
    mo.md(r"""Next, we'll need some sentences to classify. Below, we use some statements from the [International Personality Item Pool](https://ipip.ori.org/AlphabeticalItemList.htm), typically found in personality questionnaires.""")
    return


@app.cell
def _():
    sentences = [
        "I sometimes have thoughts that don't make any sense.",
        "I love to be the center of attention.",
        "I love a good fight.",
        "I shift back and forth between strong love and strong hate.",
        "I worry a lot about catching a serious illness.",
        "I would be afraid to give a speech in public.",
    ]
    return (sentences,)


@app.cell
def _(mo):
    mo.md(r"""Since we're working with pipelines this time, all we need to do is simply pass the list of sentences to our pipeline object `classifier`.""")
    return


@app.cell
def _(classifier, sentences):
    results = classifier(inputs=sentences)
    return (results,)


@app.cell
def _(pd, results, sentences):
    df = pd.DataFrame(results)
    df["sentences"] = sentences
    df
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Voilà! The `score` expresses the `label` probability, which simply is the result from applying the softmax to two output logits for each statement.

    ::lucide:message-circle-question-mark:: Examine the output. Do you agree with the classification of sentiment in each sentence? Perhaps the answer isn't all too clear.
    """
    )
    return


if __name__ == "__main__":
    app.run()