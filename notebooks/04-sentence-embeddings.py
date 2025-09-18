import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    get_input_items, set_input_items = mo.state([])
    get_input_embeddings, set_input_embeddings = mo.state([])
    return (
        get_input_embeddings,
        get_input_items,
        set_input_embeddings,
        set_input_items,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    # Sentence Embeddings
    We're going to explore semantic space!👩‍🚀🛰️

    In the first notebook on decoder models, we've used contextualized token embeddings to measure the similarity between individual words. However, in many use cases, we want to compare entire sequences of text. This is where sentence transformers come into play.

    We start by loading the relevant packages.
    """
    )
    return


@app.cell
def _(mo):
    import umap
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics.pairwise import cosine_similarity

    mo.show_code()
    return (
        MinMaxScaler,
        SentenceTransformer,
        cosine_similarity,
        go,
        np,
        pd,
        umap,
    )


@app.cell
def _(go):
    # These are two plotting functions which we will use to visualize semantic similarities. We'll skip explaining these so that we have more time to focus on natural language processing.

    def plot_similarity_matrix(similarity_matrix, labels):
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='Greys',
            zmin=0,
            zmax=1,
            showscale=False,
            hovertemplate='Text 1: %{y}<br>Text 2: %{x}<br>Similarity: %{z:.3f}<extra></extra>'
        ))

        fig.update_layout(
            template="none",
            margin=dict(l=60, r=20, t=80, b=60),
            xaxis=dict(
                tickfont=dict(color='black', size=8),
                tickangle=45,
                showgrid=False,
                zeroline=False,
                side='bottom',
                showticklabels=True,
                constrain='domain'
            ),
            yaxis=dict(
                tickfont=dict(color='black', size=8),
                showgrid=False,
                zeroline=False,
                side='left',
                showticklabels=True,
                scaleanchor='x',
                scaleratio=1
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        return fig

    def plot_semantic_space(embeddings, labels):

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=1, color='white', opacity=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter3d(
            x=embeddings[:, 0], 
            y=embeddings[:, 1], 
            z=embeddings[:, 2],
            mode='markers+text',
            marker=dict(size=6, color='black'),
            text=labels,
            textposition='middle right',
            textfont=dict(color='black', size=10),
            showlegend=False
        ))

        fig.update_layout(
            showlegend=False,
            template="none",
            scene=dict(
                aspectmode='cube',
                camera=dict(
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=-1.5, y=1.5, z=1)
                ),
                xaxis=dict(
                    nticks=7,
                    range=[-1, 1],
                    title=dict(text='e1', font=dict(color='black', size=12)),
                    showticklabels=False,
                    gridcolor='black',
                    gridwidth=0.5
                ),
                yaxis=dict(
                    nticks=7,
                    range=[-1, 1],
                    title=dict(text='e2', font=dict(color='black', size=12)),
                    showticklabels=False,
                    gridcolor='black',
                    gridwidth=0.5
                ),
                zaxis=dict(
                    nticks=7,
                    range=[-1, 1],
                    title=dict(text='e3', font=dict(color='black', size=12)),
                    showticklabels=False,
                    gridcolor='black',
                    gridwidth=0.5
                )
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )

        return fig
    return plot_semantic_space, plot_similarity_matrix


@app.cell
def _(mo):
    mo.md(r"""Next, we load a sentence transformer model from the hugging face model hub. We'll pick [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2), a light-weight, general purpose model.""")
    return


@app.cell
def _(SentenceTransformer, mo):
    with mo.status.spinner("Loading model..."):
        model = SentenceTransformer(
            model_name_or_path="sentence-transformers/all-MiniLM-L12-v2"
        )

    mo.show_code()
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""As usual, we print the model architecture to get a better understanding of what we're working with.""")
    return


@app.cell
def _(mo, model):
    print(model)
    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(r"""Now we need some text to work with. Let us again use some questionnaire items from the [International Personality Item Pool](https://ipip.ori.org/AlphabeticalItemList.htm). Below, we use 15 items with 5 items each for the personality domains extraversion, neuroticism, and conscientiousness, defined in three separate lists.""")
    return


@app.cell
def _(mo):
    # extraversion
    ext_items = [
    	"I feel comfortable around people.",
    	"I make friends easily.",
    	"I am skilled in handling social situations.",
    	"I am the life of the party.",
    	"I know how to captivate people.",
    ]

    # neuroticism
    neu_items = [
    	"I often feel blue.",
    	"I dislike myself.",
    	"I am often down in the dumps.",
    	"I have frequent mood swings.",
    	"I panic easily.",
    ]

    # conscientiousness
    con_items = [
    	"I am always prepared.",
    	"I pay attention to details.",
    	"I get chores done right away.",
    	"I carry out my plans.",
    	"I make plans and stick to them.",    
    ]

    # we also create a list of combined item texts, which we will use later
    items = (ext_items + neu_items + con_items)

    mo.show_code()
    return con_items, ext_items, items, neu_items


@app.cell
def _(mo):
    mo.md(r"""Now, we create three embedding matrices, one for each personality domain. Each row represents an item, each column a dimension in semantic space.""")
    return


@app.cell
def _(con_items, ext_items, mo, model, neu_items, np):
    with mo.status.spinner("Encoding..."):
        ext_embeddings = model.encode(ext_items)
        neu_embeddings = model.encode(neu_items)
        con_embeddings = model.encode(con_items)

    print(f"Shape of embeddings for extraversion: {ext_embeddings.shape}")

    # we also create one big embedding matrix, by stacking the individual matrices
    item_embeddings = np.vstack([ext_embeddings, neu_embeddings, con_embeddings])

    mo.show_code()
    return con_embeddings, ext_embeddings, item_embeddings, neu_embeddings


@app.cell
def _(mo):
    mo.md(r"""Next, we calculate the cosine similarity between all the item embeddings and plot the resulting matrix.""")
    return


@app.cell
def _(cosine_similarity, item_embeddings, mo):
    item_similarity_matrix = cosine_similarity(X=item_embeddings, Y=item_embeddings)
    mo.show_code()
    return (item_similarity_matrix,)


@app.cell
def _(item_similarity_matrix, items, plot_similarity_matrix):
    item_similarity_plot = plot_similarity_matrix(
        similarity_matrix=item_similarity_matrix, 
        labels=items
    )

    (
        item_similarity_plot
            .update_layout(autosize=True)
            .update_xaxes(automargin=True)
            .update_yaxes(automargin=True)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""::lucide:message-circle-question-mark:: Take some time to examine the plotted cosine similarity matrix above, by hovering over the individual cells. Do you see some patterns that might be worth discussing?""")
    return


@app.cell
def _(mo):
    mo.md(r"""Perhaps we can find a way to better represent each of the three personality traits. Picture these as three, perhaps somewhat distinct, clouds of points in 384-dimensional space. If your imagination is limited to fewer dimensions, you can also use three dimension. If we find the centroid for each cluster of item embeddings, we might get a concept vector. We can do this by averaging the item embeddings for each cluster.""")
    return


@app.cell
def _(con_embeddings, ext_embeddings, mo, neu_embeddings, np):
    ext_centroid = ext_embeddings.mean(axis=0)
    neu_centroid = neu_embeddings.mean(axis=0)
    con_centroid = con_embeddings.mean(axis=0)

    print(f"Shape of centroid for extraversion: {ext_centroid.shape}")

    # we also create one single embedding matrix, by stacking the individual matrices and add the labels for each centroid
    centroids = np.vstack([ext_centroid, neu_centroid, con_centroid])
    centroid_labels = ["EXTRAVERSION", "NEUROTICISM", "CONSCIENTIOUSNESS"]

    mo.show_code()
    return centroid_labels, centroids


@app.cell
def _(mo):
    mo.md(r"""Let's check how similar the three centroids are.""")
    return


@app.cell
def _(centroid_labels, centroids, cosine_similarity, mo, pd):
    centroid_similarity_matrix = cosine_similarity(X=centroids, Y=centroids)

    centroid_similarity_df = pd.DataFrame(
        data=centroid_similarity_matrix, 
        columns=centroid_labels, 
        index=centroid_labels
    )


    mo.output.append(
        mo.vstack([
            mo.ui.table(
                data=centroid_similarity_df,
                format_mapping={
                    centroid_labels[0]: "{:.2f}".format,
                    centroid_labels[1]: "{:.2f}".format,
                    centroid_labels[2]: "{:.2f}".format,
                }
            ),
            mo.show_code()
        ])
    )
    return


@app.cell
def _(mo):
    mo.md(r"""Next, we would like to find some way to intuitively test our personality centroids. Below, we define a simple user interface, which lets us enter new items on the fly, which we can then use to compare against our centroids.""")
    return


@app.cell
def _(mo):
    item_input = mo.ui.text_area(
        label="Add Items to Semantic Space",
        rows=9,
        value=(
            "I recharge by being around people\n"
            "I speak up in group discussions\n"
            "I seek out social gatherings\n"
            "I worry about things going wrong\n"
            "My mood shifts frequently\n"
            "I feel overwhelmed by stress\n"
            "I follow through on my commitments\n"
            "I plan ahead for important tasks\n"
            "I overthink my grocery list punctuation\n"
        ),
        full_width=True
    )
    run_button = mo.ui.run_button(label="Enter", full_width=True)

    mo.vstack([mo.show_code(), item_input, run_button])
    return item_input, run_button


@app.cell
def _(
    centroids,
    cosine_similarity,
    get_input_items,
    item_input,
    mo,
    model,
    run_button,
    set_input_embeddings,
    set_input_items,
):
    # the following code is only executed when the button above is pressed
    if run_button.value and len(item_input.value) > 0:

        # we split the values from the text area into sentences, line by line
        item_input_values = item_input.value.split("\n")
        item_input_values = [x.strip() for x in item_input_values if len(x) > 0]
        set_input_items(item_input_values)

        # we obtain embeddings for the items in the textarea and calculate cosine similarity
        input_embeddings = model.encode(get_input_items())
        set_input_embeddings(input_embeddings)

        input_similarity_matrix = cosine_similarity(X=input_embeddings, Y=centroids)
    mo.show_code()
    return input_similarity_matrix, item_input_values


@app.cell
def _(mo):
    mo.md(r"""Let's examine the semantic similarity between the items we have entered in the user interface and our construct centroids.""")
    return


@app.cell
def _(
    centroid_labels,
    input_similarity_matrix,
    item_input,
    item_input_values,
    mo,
    pd,
    run_button,
):
    # this block is less important: we only convert our similarity matrix to a data frame
    # and find a way to visualize it nicely
    if run_button.value and len(item_input.value) > 0:

        input_similarity_df = pd.DataFrame(
            data=input_similarity_matrix, 
            columns=centroid_labels, 
            index=item_input_values
        )

        mo.output.append(
            mo.ui.table(
                data=input_similarity_df,
                format_mapping={
                    centroid_labels[0]: "{:.2f}".format,
                    centroid_labels[1]: "{:.2f}".format,
                    centroid_labels[2]: "{:.2f}".format,
                }
            )
        )
    return


@app.cell
def _(mo):
    mo.md(r"""We should see that items expressing extraverted statements show higher associations with the `EXTRAVERSION` centroid, etc.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We can try to visualize semantic space, by reducing it down to 3 dimensions from 384. You probably already are familiar with some dimensionality reduction techniques like Principal Component Analysis (PCA). However, in our case, Uniform Manifold Approximation and Projection (UMAP) as several advantages. First, PCA is a linear transformation, while UMAP preserves non-linear structures. Second, UMAP keeps semantically similar embeddings closer together after dimensionality reduction and therefore does a better job in preserving cluster neighborhood.

    We go on by reducing semantic space down to 3 dimensions, based on our initial set of items. Then, we use the same UMAP model to reduce the item embeddings from our textarea input. Finally, we combine or centroids and the embeddings from our text area input to a dataset, which we visualize in 3D space.
    """
    )
    return


@app.cell
def _(
    MinMaxScaler,
    centroid_labels,
    centroids,
    get_input_embeddings,
    get_input_items,
    item_embeddings,
    mo,
    np,
    plot_semantic_space,
    umap,
):
    mo.stop(len(get_input_items()) < 1)

    with mo.status.spinner("Dimensionality reduction..."):

        # we define our UMAP model
        reducer = umap.UMAP(
            n_components=3, # the number of dimensions we want to arrive at
            n_neighbors=5, # smaller numbers are better att preserving fine-grained relationships
            random_state=42, # for reproducibility
        )

        # we first fit the UMAP model from above
        embeddings_reduced = reducer.fit_transform(item_embeddings)

        # we now reduce the embeddings from our textarea input and our construct centroids
        input_embeddings_reduced = reducer.transform(get_input_embeddings())
        centroids_reduced = reducer.transform(centroids)

        # we add them together to one big matrix
        space_embeddings = np.vstack([centroids_reduced, input_embeddings_reduced])

        # we use min-max-scaling to scale the loadings to a more interpretable range
        scaler = MinMaxScaler(feature_range=(-1, 1))
        space_embeddings_scaled = scaler.fit_transform(space_embeddings)

        space_labels = centroid_labels + get_input_items()

        semantic_space_plot = plot_semantic_space(
            embeddings=space_embeddings_scaled,
            labels=space_labels
        )

        mo.output.append(
            mo.vstack([
                mo.show_code(),
                semantic_space_plot,
                mo.md("Drag mouse to zoom, CTRL + mouse to pan, and SHIFT + mouse to rotate the plot.")
            ])
        )

    return


if __name__ == "__main__":
    app.run()
