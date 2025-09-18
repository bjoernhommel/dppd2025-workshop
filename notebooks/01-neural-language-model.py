import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    # marimo specific code: you can ignore it for now

    get_step, set_step = mo.state(1)
    get_results, set_results = mo.state([])

    get_initial_params, set_initial_params = mo.state(None)
    get_current_params, set_current_params = mo.state(None)

    def add_result(data):
        set_results(lambda x: x + [data])
    return (
        add_result,
        get_initial_params,
        get_results,
        get_step,
        set_current_params,
        set_initial_params,
        set_results,
        set_step,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # Exercise #1 - A Neural Probabilistic Language Model
    In this exercise, we re-construct a simple language model based on [Bengio et al. (2003)](https://www.jmlr.org/papers/v3/bengio03a.html) to demonstrate the fundamental building blocks of modern LLMs. This code was adapted from [Sihyung Park's Blogpost](https://naturale0.github.io/2021/02/04/Understanding-Neural-Probabilistic-Language-Model).

    Let's examine this notebook in **app mode** and ignore most of  the code.
    """
    )
    return


@app.cell
def _(mo):
    import copy
    import nltk
    import string
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import altair as alt
    import numpy as np
    import pandas as pd

    with mo.status.spinner("Loading gutenberg project"):
        nltk.download('gutenberg')
        from nltk.corpus import gutenberg
    return copy, gutenberg, nn, optim, pd, string, torch


@app.cell
def _(gutenberg, mo):
    corpus_preview = gutenberg.raw('carroll-alice.txt')[:800] + "..."

    mo.vstack([
        mo.md("## Preview Training Corpus"),
        mo.md('''
        Let\'s train a very simple language model. For demonstration purposes, we obtain a small text corpus from the [Gutenberg Project](https://www.gutenberg.org/ebooks/), namely "Alice's Adventures in Wonderland" by Lewis Carroll.
        '''),
        mo.callout(corpus_preview)
    ])
    return


@app.cell
def _(gutenberg, string):
    def clean_corpus(corpus):
        corpus = corpus.translate(str.maketrans('', '', string.punctuation))
        corpus = ' '.join(corpus.lower().split())
        return corpus

    corpus = clean_corpus(gutenberg.raw('carroll-alice.txt').lower())
    return (corpus,)


@app.cell
def _(corpus):
    words = corpus.split()
    vocabulary = {word.lower(): i for i, word in enumerate(set(words))}
    vocab_size = len(vocabulary)
    return vocab_size, vocabulary, words


@app.cell
def _(corpus, vocabulary):
    def tokenize(corpus, vocabulary):
        words = corpus.lower().split()
        return [vocabulary[word] for word in words if word in vocabulary]

    def detokenize(tokens, vocabulary):
        tokens = tokens if isinstance(tokens, list) else [tokens]
        id_to_word = {v: k for k, v in vocabulary.items()}
        words = [id_to_word[token] for token in tokens if token in id_to_word]
        return ' '.join(words)

    tokenized_corpus = tokenize(corpus, vocabulary)
    return detokenize, tokenized_corpus


@app.cell
def _(mo, pd, tokenized_corpus, vocabulary, words):
    mo.vstack([
        mo.md("## Cleaning and Tokenizing the Corpus"),
        mo.md('''
        After performing some very basic cleaning of the text (e.g., removing punctuations), we create a vocabulary of unique words in the corpus. We further convert all the words in the vocabulary and in the corpus to numeric representations (i.e., tokens).
        '''),
        mo.hstack([
            mo.vstack([mo.md("**Vocabulary**"), vocabulary]), 
            mo.vstack([mo.md("**Corpus (tokenized)**"), tokenized_corpus]),    
            mo.vstack([mo.md("**Distribution of Words in Corpus**"), pd.Series(words).value_counts().plot()]),    
        ])
    ])
    return


@app.cell
def _(mo):
    embedding_dim = mo.ui.number(
        label="Embedding dimensions",
        start=1,
        value=100
    )

    hidden_dim = mo.ui.number(
        label="Hidden layer dimensions",
        start=1,
        value=128
    )

    context_size = mo.ui.number(
        label="Context size",
        start=1,
        value=3
    )
    return context_size, embedding_dim, hidden_dim


@app.cell
def _(mo):
    mo.md(r"""We define a very basic language model, similar to the one proposed by [Bengio et al. (2003)](https://www.jmlr.org/papers/v3/bengio03a.html).""")
    return


@app.cell
def _(mo, nn, torch):
    class NeuralProbabilisticLanguageModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim):
            super(NeuralProbabilisticLanguageModel, self).__init__()

            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
            self.tanh = nn.Tanh()
            self.linear2 = nn.Linear(hidden_dim, vocab_size)

        def forward(self, inputs):
            embeds = self.embeddings(inputs).view((1, -1))
            out = self.linear1(embeds)
            out = self.tanh(out)
            out = self.linear2(out)
            log_probs = torch.log_softmax(out, dim=1)
            return log_probs

    mo.show_code()
    return (NeuralProbabilisticLanguageModel,)


@app.cell
def _(mo):
    mo.md(
        r"""
    The model extends the neural network base class from pytorch (`nn.Module`), but you don't need to worry about this (or `super()` in this tutorial).

    The `__init__` functions shows us the architecture of the model. It consists of a `embeddings` layer, which is connected to a `linear` (hidden) layer, using our activation function `self.Tanh()`. The hidden layer, in turn, is connected to the projection layer `self.linear2`. 

    We also define what is happening in a prediction step, defined in the `forward()` function. The model input is converted to embeddings, which are then flattened to a 1D vector, using `.view((1, -1))`. After passing this vector through the network, we finally obtain the log probabilities for the entire vocabulary using `log_softmax()`.

    That's it! We just built an entire language model from scratch!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""Now, before initializing the model, we decide on the configuration of the language model. We use a simple user interface for this.""")
    return


@app.cell
def _(context_size, embedding_dim, hidden_dim, mo):
    mo.vstack([
        mo.md("## Model Configuration"),
        mo.hstack([
            mo.vstack([
                embedding_dim, 
                hidden_dim, 
                context_size
            ]),
            mo.md("Configure model hyperparameters")        
        ])
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""Now, we initialize the model using the inputs from the user interface.""")
    return


@app.cell
def _(
    NeuralProbabilisticLanguageModel,
    context_size,
    embedding_dim,
    hidden_dim,
    mo,
    nn,
    optim,
    vocab_size,
):
    model = NeuralProbabilisticLanguageModel(
        vocab_size, 
        embedding_dim.value, 
        context_size.value, 
        hidden_dim.value
    )

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    mo.show_code()
    return loss_function, model, optimizer


@app.cell
def _(mo):
    mo.md(r"""Let's print the model architecture to confirm that everything makes sense.""")
    return


@app.cell
def _(model):
    model
    return


@app.cell
def _(copy, get_initial_params, model, set_initial_params):
    if get_initial_params() is None:
        set_initial_params({
            "embeddings": copy.deepcopy(model.embeddings.weight.data.numpy()),
            "linear1_weights": copy.deepcopy(model.linear1.weight.data.numpy()),
            "linear1_biases": copy.deepcopy(model.linear1.bias.data.numpy()),
            "linear2_weights": copy.deepcopy(model.linear2.weight.data.numpy()),
            "linear2_biases": copy.deepcopy(model.linear2.bias.data.numpy()),
        })
    return


@app.cell
def _(context_size, tokenized_corpus):
    training_data = []
    for j in range(len(tokenized_corpus) - context_size.value):
        training_data.append({
            "context": tokenized_corpus[j:j + context_size.value],
            "target": tokenized_corpus[j + context_size.value]
        })
    return (training_data,)


@app.cell
def _(mo):
    mo.md(r"""We now define another user interface, which we can use to train the model manually.""")
    return


@app.cell
def _(get_step, mo, training_data):
    start_step_input = mo.ui.number(
        label="Start Training at Step",
        value=get_step(),
        start=1, 
        stop=len(training_data)
    )

    step_count_input = mo.ui.number(
        label="Training Number of Steps",
        value=10,
        start=1
    )

    epoch_count_input = mo.ui.number(
        label="Training Number of Epochs",
        value=1,
        start=1
    )

    train_model_button = mo.ui.run_button(label="Train Model")
    reset_model_button = mo.ui.run_button(label="Reset Model")

    mo.vstack([
        mo.md("## Training Controls"),
        start_step_input,
        step_count_input,
        epoch_count_input,
        mo.hstack([train_model_button, reset_model_button], justify="start")
    ])
    return (
        epoch_count_input,
        reset_model_button,
        start_step_input,
        step_count_input,
        train_model_button,
    )


@app.cell
def _(
    detokenize,
    mo,
    start_step_input,
    tokenized_corpus,
    training_data,
    vocabulary,
):
    current_step = start_step_input.value

    history_tokens = tokenized_corpus[0:current_step-1]
    history = detokenize(history_tokens, vocabulary)

    preview_context_tokens = training_data[current_step-1]['context']
    preview_context = detokenize(preview_context_tokens, vocabulary)

    preview_target_tokens = training_data[current_step-1]['target']
    preview_target = detokenize(preview_target_tokens, vocabulary)

    mo.vstack([    
        mo.md("## Training Preview"),
        mo.vstack([mo.md("**History**"), mo.callout(history if history else "")]),
        mo.hstack([
            mo.vstack([
                mo.md(f'**Context**: "{preview_context}"'),
                mo.md(f'**Context tokens**: {preview_context_tokens}')
            ]),
            mo.vstack([
                mo.md(f'**Target**: "{preview_target}"'),
                mo.md(f'**Target token**: {preview_target_tokens}')
            ]),
        ]),
    ])
    return


@app.cell
def _(
    add_result,
    copy,
    epoch_count_input,
    loss_function,
    mo,
    model,
    optimizer,
    set_current_params,
    set_step,
    start_step_input,
    step_count_input,
    torch,
    train_model_button,
    training_data,
):
    if train_model_button.value:

        training_epochs = range(epoch_count_input.value)

        for epoch in mo.status.progress_bar(training_epochs):
            from_step = (start_step_input.value - 1)
            to_step = from_step + step_count_input.value

            training_steps = range(from_step, min(to_step, len(training_data)))

            for step in mo.status.progress_bar(training_steps):
                context_tensor = torch.tensor(training_data[step]['context'], dtype=torch.long)
                target_tensor = torch.tensor([training_data[step]['target']], dtype=torch.long)

                log_probs = model(context_tensor)
                loss = loss_function(log_probs, target_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                add_result({
                    'step': step + 1, 
                    'loss': loss.item()
                })

        current_params = {
            "embeddings": copy.deepcopy(model.embeddings.weight.data.numpy()),
            "linear1_weights": copy.deepcopy(model.linear1.weight.data.numpy()),
            "linear1_biases": copy.deepcopy(model.linear1.bias.data.numpy()),
            "linear2_weights": copy.deepcopy(model.linear2.weight.data.numpy()),
            "linear2_biases": copy.deepcopy(model.linear2.bias.data.numpy()),
        }

        set_current_params(current_params)
        set_step(to_step + 1)
    return


@app.cell
def _(get_results, mo, pd):
    training_history_plot = pd.DataFrame() 
    if get_results():
        training_history_plot = pd.DataFrame(get_results()).reset_index().plot(x="index", y="loss")

    mo.vstack([
        mo.md("## Training History"),
        mo.hstack([        
            pd.DataFrame(get_results()),
            training_history_plot
        ])
    ])
    return


@app.cell
def _(
    copy,
    mo,
    model,
    reset_model_button,
    set_current_params,
    set_initial_params,
    set_results,
    set_step,
):
    if reset_model_button.value:

        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        set_results([])
        set_step(1)

        initial_params = {
            "embeddings": copy.deepcopy(model.embeddings.weight.data.numpy()),
            "linear1_weights": copy.deepcopy(model.linear1.weight.data.numpy()),
            "linear1_biases": copy.deepcopy(model.linear1.bias.data.numpy()),
            "linear2_weights": copy.deepcopy(model.linear2.weight.data.numpy()),
            "linear2_biases": copy.deepcopy(model.linear2.bias.data.numpy()),
        }
        set_initial_params(initial_params)
        set_current_params(None)

        mo.output.replace(
            mo.vstack([mo.md("🔄 Model has been reset! Parameters stored for comparison.")])
        )
    return


if __name__ == "__main__":
    app.run()