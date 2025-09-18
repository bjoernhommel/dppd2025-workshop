import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # Masked Language Modeling

    In this notebook, we will use a small decoder model to solve fill-mask tasks, also known as masked language modeling. We'll also learn some basics about tokenizers and token embeddings.

    Before we start, we'll load some required python packages.
    """
    )
    return


@app.cell
def _(mo):
    from copy import copy
    import torch
    import pandas as pd
    import torch.nn.functional as F
    mo.show_code()
    return F, copy, pd, torch


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Loading the Tokenizer
    We first load the tokenizer for a desired model (i.e., [distilbert](https://huggingface.co/distilbert/distilbert-base-uncased)) from the Hugging Face Model Hub. Remember that a tokenizer is used to convert sequences of text to linguistics units (i.e., "tokens"), which then can be used by the transformer model.
    """
    )
    return


@app.cell
def _(mo):
    with mo.status.spinner("Loading tokenizer..."):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    mo.show_code()
    return (tokenizer,)


@app.cell
def _(mo):
    mo.md("""Let's start by examining some basic features of the tokenizer.""")
    return


@app.cell
def _(mo, tokenizer):
    print(f'Tokenizer name: {tokenizer.__class__.__name__}')
    print(f'Vocabulary size: {tokenizer.vocab_size}')
    print(f'Special tokens: {tokenizer.all_special_tokens}')
    print(f'Special ids: {tokenizer.all_special_ids}')
    print(f'Maximum sequence length: {tokenizer.model_max_length}')

    mo.show_code(position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""Now let's see how the tokenizer segments a simple sentence.""")
    return


@app.cell
def _(mo, tokenizer):
    sentence = "Let\'s go camping Ѧ"
    print(f"Tokens: {tokenizer.tokenize(sentence)}")
    print(f"Token IDs: {tokenizer.encode(sentence)}")
    mo.show_code(position="above")
    return (sentence,)


@app.cell
def _(mo):
    mo.md(
        r"""
    Here, we can see that the character `Ѧ` was converted to a special token `[UNK]` for tokens which are out-of-vocabulary. We also see that the tokenizer used the `[CLS]` with id 100 at the start of the sentence and `[SEP]` with id 102 to denote the end of the sentence.

    We can expand the vocabulary of our tokenizer by adding the unknown character `Ѧ` to it.
    """
    )
    return


@app.cell
def _(copy, mo, sentence, tokenizer):
    # Note: You can usually ignore the step of copying your tokenizer. This is only necessary in this marimo-notebook. 
    tokenizer_copy = copy(tokenizer)

    tokenizer_copy.add_tokens("Ѧ")
    print(f"Tokens: {tokenizer_copy.tokenize(sentence)}")
    print(f"Token IDs: {tokenizer_copy.encode(sentence)}")

    mo.show_code(position="above")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Excellent, the new token is now in our vocabulary. 


    **::lucide:triangle-alert:: Beware when adding new tokens**: Models will fail to predict tokens that were added after training. You will need to re-train the model.

    Next, let us decode a tokenized sentence to convert it back to text from numeric input id's.
    """
    )
    return


@app.cell
def _(mo, tokenizer):
    print(f"Decoded sentence: {tokenizer.decode([
        101, 2298, 2012, 2017, 1010, 21933, 4667, 2115, 2034, 6251, 999, 2115, 3008, 2097, 2022, 4013, 8630, 999, 102
    ])}")
    mo.show_code(position="above")
    return


@app.cell
def _(mo):
    mo.md(r"""Next, we'll encode multiple sentences which we'll make use of later.""")
    return


@app.cell(hide_code=True)
def _(mo, tokenizer):
    sentences1 = [
        "We are all just monkeys with money and [MASK].", # we mask "guns"
        "Got a head full of lightning, a [MASK] full of rain.", # we mask "hat"
        "The [MASK] was like electric sugar.", # we mask "music"
        "There ain't no [MASK] there's just god when he's drunk", # we mask "devil"
    ]

    with mo.status.spinner():
        tokenizer_output1 = tokenizer(
            text=sentences1,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        for i in range(len(tokenizer_output1['input_ids'])):
            print(f"Sentence #{i+1}")
            print(f"Text: {sentences1[i]}")
            print(f"Token ids: {tokenizer_output1['input_ids'][i]}")
            print(f"Attention mask: {tokenizer_output1['attention_mask'][i]}")
            print("\n")

    mo.show_code(position="above")
    return sentences1, tokenizer_output1


@app.cell
def _(mo):
    mo.md(
        r"""
    Note that `padding=True` has ensured that the lists of token ids in our output all have the same length by adding `[PAD]` with the token id 0 to shorter sentences. This is important, as the model expects tensors in the same shape when training or predicting batches of sentences. As seen in the `attention_mask`, these tokens will not influence our predictions.

    The `truncation=True` flag prevents us from exceeding the `model_max_length`.

    As you can see, the `[MASK]` token is the special token id 103 from earlier. We will try to predict the missing token. This is often referred to as a **fill-mask** task.

    But first, we need to load the model.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Loading the Encoder Model

    Great, so now that we have gotten some basic understanding for the tokenizer, we can proceed by loading the actual transformer model, which we are going to use.
    """
    )
    return


@app.cell
def _(mo):
    with mo.status.spinner("Loading model..."):
        from transformers import AutoModelForMaskedLM

        model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

    mo.show_code()
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""Let's briefly examine the model object.""")
    return


@app.cell
def _(mo, model):
    print(model)
    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(r"""::lucide:message-circle-question-mark:: What information do we get about the architecture of this model?""")
    return


@app.cell
def _(mo):
    mo.md(r"""Now, we can use the model to predict the masked tokens by passing the ids for the tokens and the attention mask to the model.""")
    return


@app.cell
def _(mo, model, tokenizer_output1, torch):
    with mo.status.spinner():
        with torch.no_grad():
            outputs1 = model(
                input_ids=tokenizer_output1['input_ids'],
                attention_mask=tokenizer_output1['attention_mask']
            )
            logits = outputs1.logits

    mo.show_code()
    return (logits,)


@app.cell
def _(mo):
    mo.md(
        r"""
    We extract the predicted `logits` from the model output. The wrapper `with torch.no_grad()`, prevents the computation of gradients, which are only needed for backpropagation (i.e., training the model), thus saving memory.

    The dimensions of the `logits` tensor are as follows:
    """
    )
    return


@app.cell
def _(logits, mo):
    print(logits.shape)
    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(r"""::lucide:message-circle-question-mark:: Can you explain the shape of the tensor?""")
    return


@app.cell
def _(logits, mo, sentences1, tokenizer, tokenizer_output1, torch):
    top_k = 7

    # We loop through the sentences...
    with mo.status.spinner():
        for j, sentence_ids in enumerate(tokenizer_output1['input_ids']):

            # We find the position of the first mask token (although there is only one).
            mask_positions = torch.where(sentence_ids == tokenizer.mask_token_id)[0]

            # We loop through the mask tokens (again, only 1)
            for pos in mask_positions:

                mask_logits = logits[j, pos]
                # The softmax function converts logits to probabilities
                mask_probs = torch.softmax(mask_logits, dim=0)

                # the topk function gives us the k = 7 most probable tokens
                top_k_ids = torch.topk(mask_logits, k=top_k).indices
                top_k_probs, top_k_ids = torch.topk(mask_probs, k=top_k)
                top_k_tokens = tokenizer.convert_ids_to_tokens(top_k_ids)

                print(f"Sentence #{j+1}")
                print(f"Text: {sentences1[j]}")        
                print(f"Predictions: {top_k_tokens}")
                print(f"Probabilities: {top_k_probs}")
            print("\n")

    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(r"""Hm, looks like our model isn't quite as poetic as Tom Waits.""")
    return


@app.cell
def _(mo):
    mo.md(r"""::lucide:message-circle-question-mark:: What is the probability for the first target word in the first sentence (i.e., "guns")?""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Working with Embeddings
    As you may recall from the Transformer model architecture, the input sequence is represented as a dense matrix that encodes semantic information about the tokens in the sequence (i.e., Token Embeddings). In contrast to older, simpler embedding models (e.g., word2vec), these embeddings are positionally encoded and contextualized.

    Let's obtain embeddings for the following two sentences.
    """
    )
    return


@app.cell
def _(mo, model, tokenizer, torch):
    sentences2 = [
        "I am the life of the party.",
        "I vote for the same party in every election."
    ]

    # Again, we tokenize the sentences and obtain the model output.
    with mo.status.spinner():
        tokenizer_output2 = tokenizer(
            text=sentences2,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

    with torch.no_grad():
        outputs2 = model(
            input_ids=tokenizer_output2['input_ids'],
            attention_mask=tokenizer_output2['attention_mask'],
            output_hidden_states=True
        )
    return outputs2, sentences2, tokenizer_output2


@app.cell
def _(mo):
    mo.md(r"""We'll find the embeddings nested under `hidden_states[0]` in the output object (`outputs2`).""")
    return


@app.cell
def _(mo, outputs2):
    print(len(outputs2.hidden_states))
    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The length of the hidden state may seem odd at first glance, but remember that Transformer models have multiple layers. This model has 12 layers in addition to the embedding layer.

    We want to use the top layer for most tasks, as this contains rich contextual information which was processed by all preceding layers.
    """
    )
    return


@app.cell
def _(mo, outputs2):
    print(outputs2.hidden_states[-1].shape)
    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Again, the shape tells us that we have 2 sequences of text with 12 tokens each. The last number denotes the model dimensionality, i.e., the size of each embedding vector.

    This is a perfect opportunity to see how Transformers handle *homonyms* (i.e., the same word having two different meanings). In the two sentences above, the word "party" has two completely different meanings.

    Let us find the embeddings for "party" in each sentence.
    """
    )
    return


@app.cell
def _(mo, outputs2, pd, tokenizer, tokenizer_output2, torch):
    party_token = tokenizer.encode("party")[1]
    print(f'"Party" Token Id: {party_token}')

    party_pos1 = torch.where(tokenizer_output2.input_ids[0] == party_token) # Position of "Party" in first sentence
    party_pos2 = torch.where(tokenizer_output2.input_ids[1] == party_token) # Position of "Party" in second sentence

    party_embeddings1 = outputs2.hidden_states[-1][0, party_pos1, :] # Embeddings for "Party" in first sentence
    party_embeddings2 = outputs2.hidden_states[-1][1, party_pos2, :] # Embeddings for "Party" in second sentence


    party_embeddings = pd.DataFrame(
        data=torch.cat([party_embeddings1, party_embeddings2], dim=0).detach().numpy(), 
        columns=[f"e{x}" for x in range(party_embeddings1.shape[-1])]
    )


    mo.show_code()
    return party_embeddings, party_embeddings1, party_embeddings2


@app.cell
def _(party_embeddings):
    party_embeddings
    return


@app.cell
def _(mo):
    mo.md(r"""Cool, the embeddings for the same word "party" seem to vary quite a bit, depending on their context. We use cosine similarity to quantify the effect.""")
    return


@app.cell
def _(F, mo, party_embeddings1, party_embeddings2, sentences2):
    similarity = F.cosine_similarity(party_embeddings1, party_embeddings2, dim=-1)
    print(f'''
        Similarity of "party" in {sentences2}: {similarity.item()}
    ''')
    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(r"""::lucide:message-circle-question-mark:: You might be wondering if the magnitude of the cosine similarity indicates a rather dissimilar meaning between the homonyms? To put it into perspective, add and compare another use of the word "party", in a third sentence!""")
    return


if __name__ == "__main__":
    app.run()
