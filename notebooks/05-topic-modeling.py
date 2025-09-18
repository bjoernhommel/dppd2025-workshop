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
        r"""
    # Topic Modeling with BERTopic

    In this exercise, we'll briefly explore topic modeling - a suite of exploratory methods to extract and group conceptually similar documents. We'll use BERTopic for this, which is a nice wrapper package for the statistical methods that do the actual lifting in the background. BERTopic is easy to use and modular, which allows for great flexibility.

    We start by importing Python packages.
    """
    )
    return


@app.cell
def _(mo):
    import string
    import pandas as pd
    import numpy as np
    from umap import UMAP
    from hdbscan import HDBSCAN
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    mo.show_code()
    return (
        BERTopic,
        HDBSCAN,
        SentenceTransformer,
        UMAP,
        cosine_similarity,
        np,
        pd,
        string,
    )


@app.cell
def _(mo):
    mo.md(r"""As usual, we will need some data, ideally a somewhat larger collection of unstructured documents. Documents in topic modeling usually refer to anything from short sentences to longer articles, or even summaries of books. Again, we use the [International Personality Item Pool](https://ipip.ori.org/AlphabeticalItemList.htm), but this time, let us load the entire collection of items into one large data frame.""")
    return


@app.cell
def _(mo, pd):
    ipip_url = "https://ipip.ori.org/TedoneItemAssignmentTable30APR21.xlsx"
    df = pd.read_excel(ipip_url)

    mo.show_code()
    return (df,)


@app.cell
def _(df, mo):
    mo.ui.table(df)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    As you can see, the dataset is already labeled by the personality trait (`label` column) the item (`text` column) is supposed to measure. Let's pretend that the label isn't there and see how topic modeling groups the items instead.

    Next, we extract all the items into a list, do a little bit of text cleaning and remove duplicates.
    """
    )
    return


@app.cell
def _(df):
    items = (
        df['text']
            .str.lower()
            .str.strip()
            .unique()
            .tolist()
        )
    return (items,)


@app.cell
def _(mo):
    mo.md(
        r"""
    BERTopic approaches topic modeling by stacking up to six modules:

    1. **Embedding**: Converts documents into numerical vectors to capture their semantic meaning and enable mathematical comparison.
    2. **Dimensionality Reduction**: Reduces vector complexity to make clustering computationally feasible.
    3. **Clustering**: Groups semantically similar documents together to form coherent topics.
    4. **Vectorization**: Extracts terms and phrases from each cluster to identify the vocabulary that characterizes each topic.
    5. **Weight scheme**: Calculates which terms are most distinctive and important for each specific topic compared to others.
    6. **Representation**: Refines and improves topic labels to make them more coherent and human-interpretable for end users.

    As some of these modules are optional, we only use the first three components (embedding, dimensionality reduction, and clustering) in this brief tutorial.

    Again, we use a sentence transformer as our embedding model ([all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)) and UMAP as our dimensionality reduction technique.

    For clustering, we use [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html). In contrast to other clustering techniques, such as k-means clustering, which is more established in the behavioral sciences, HDBSCAN has a couple of advantages for our use case. First, HDBSCAN automatically determines the optimal number of clusters, while k-means requires you to specify k beforehand. Second, HDBSCAN doesn't insist on assigning each and every data point to a cluster, while k-means forces outliers into clusters.

    We go on by defining these three modules and pass them to the BERTopic wrapper.
    """
    )
    return


@app.cell
def _(BERTopic, HDBSCAN, SentenceTransformer, UMAP, mo):
    with mo.status.spinner():
        embedding_model = SentenceTransformer("all-MiniLM-L12-v2")

        umap_model = UMAP(
            n_neighbors=15, # lower values preserve local structures, larger values preserve global structure
            n_components=5, 
            min_dist=0.0, 
            metric='cosine', 
            random_state=42
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=10, 
            metric='euclidean', # works well in lower dimensions
            prediction_data=True
        )

        topic_model = BERTopic(
          embedding_model=embedding_model,
          umap_model=umap_model,
          hdbscan_model=hdbscan_model,
          verbose=True
        )

    mo.show_code()
    return embedding_model, topic_model


@app.cell
def _(mo):
    mo.md(r"""Now, we are ready to fit our topic model to the items.""")
    return


@app.cell
def _(items, mo, topic_model):
    with mo.status.spinner():
        topics, probs = topic_model.fit_transform(items)

    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The default weight scheme used in BERTopic is c-TF-IDF (Class-based Term Frequency-Inverse Document Frequency) and extracts representative words to label each cluster. This often results in somewhat unintuitive cluster names. We could use the *Representation* module to come up with better cluster labels, but this usually requires loading an additional labeling model or an API key to proprietary models.

    Instead, let us use [semantic search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) using our sentence transformer model, which we have already loaded. Our strategy here is as follows:

    1. Get all the unique words used throughout the item corpus
    2. Obtain embeddings for these words
    3. Average the item embeddings for each cluster
    4. Compare the averaged cluster centroid embeddings with each word embedding, using cosine similarity
    5. Nominate the word with the highest similarity to the cluster centroid as the new cluster label

    First, we get our word list and the corresponding embeddings:
    """
    )
    return


@app.cell
def _(embedding_model, items, mo, string):
    def get_wordlist(items):
        translator = str.maketrans('', '', string.punctuation)
        items_as_longstring = " ".join(items).translate(translator)
        unique_words = list(set(items_as_longstring.split(" ")))
        return unique_words

    words = get_wordlist(items)
    word_embeddings = embedding_model.encode(words)

    mo.show_code()
    return word_embeddings, words


@app.cell
def _(mo):
    mo.md(r"""Now, we average the item embeddings for each cluster, compare them to the word embeddings and retain the word index for the highest similarity. Remember that HDBSCAN handles outliers by assigning them to a residual cluster? We label these outliers as `"[unclassified]"`.""")
    return


@app.cell
def _(
    cosine_similarity,
    embedding_model,
    items,
    mo,
    np,
    topic_model,
    word_embeddings,
    words,
):
    items_by_cluster = topic_model.get_document_info(items).groupby("Topic").agg(list)

    topic_labels_dict = {}
    with mo.status.progress_bar(total=items_by_cluster.shape[0]) as bar:
        for cluster_index, cluster in items_by_cluster.iterrows():
            if cluster_index == -1:
                topic_labels_dict[cluster_index] = "[unclassified]"
                continue
            cluster_centroid = embedding_model.encode(cluster.Document).mean(axis=0).reshape(1, -1)
            word_to_cluster_sim = cosine_similarity(word_embeddings, cluster_centroid)
            word_index = np.argmax(word_to_cluster_sim)
            topic_labels_dict[cluster_index] = words[word_index]
            bar.update()

    mo.show_code()
    return (topic_labels_dict,)


@app.cell
def _(mo):
    mo.md(r"""Let's take a look at our new cluster labels.""")
    return


@app.cell
def _(topic_labels_dict):
    print(topic_labels_dict)
    return


@app.cell
def _(mo):
    mo.md(r"""Finally, we simply update our clusters with the new labels.""")
    return


@app.cell
def _(topic_labels_dict, topic_model):
    topic_model.set_topic_labels(topic_labels_dict)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Let's visualize the cluster solution on a 2D plane. Note that in reality, UMAP has reduced the semantic space to 5 dimensions, which we won't be able to see because of our primitive monkey brains. This is why you will perceive some data points as "further apart" from the rest of their cluster than they actually are in the higher-dimensional space.

    So, fellow troglodytes, behold: The semantic landscape of personality items
    """
    )
    return


@app.cell
def _(items, mo, topic_model):
    plot = topic_model.visualize_documents(
        docs=items, 
        custom_labels=True,
        title='<b>Conceptual Item Clusters</b>', 
        width=1000, 
        height=570,
    )

    mo.show_code()
    return (plot,)


@app.cell
def _(plot):
    plot
    return


@app.cell
def _(mo):
    mo.md(r"""::lucide:message-circle-question-mark:: Look at the clusters and discuss if these make sense. Perhaps you'll notice something we've already discussed in the sentence embedding-exercise?""")
    return


@app.cell
def _(mo):
    mo.md(r"""Below, you'll find the cluster solution exported as a table. As a bonus, we get the membership probability for each item to its assigned cluster.""")
    return


@app.cell
def _(items, mo, topic_model):
    columns = ['Topic', 'CustomName', 'Document', 'Probability']

    table = topic_model.get_document_info(items)[columns]

    mo.show_code()
    return (table,)


@app.cell
def _(mo, table):
    mo.ui.table(table)
    return


if __name__ == "__main__":
    app.run()
