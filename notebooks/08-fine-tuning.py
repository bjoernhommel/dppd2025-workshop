import marimo

__generated_with = "0.15.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    get_trained_regressive_sentiment, set_trained_regressive_sentiment = mo.state(None)
    get_regressive_sentiment_history, set_regressive_sentiment_history = mo.state(None)

    get_trained_regressive_desirability, set_trained_regressive_desirability = mo.state(None)
    get_regressive_desirability_history, set_regressive_desirability_history = mo.state(None)
    return (
        get_regressive_desirability_history,
        get_regressive_sentiment_history,
        get_trained_regressive_desirability,
        get_trained_regressive_sentiment,
        set_regressive_desirability_history,
        set_regressive_sentiment_history,
        set_trained_regressive_desirability,
        set_trained_regressive_sentiment,
    )


@app.cell
def _():
    import asyncio
    import os

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # Fine-tuning and domain adaptation

    In this exercise, we will fine-tune a model to automatically rate the social desirability of survey items. We'll partially reproduce the methodology described in [Hommel (2023)](https://www.sciencedirect.com/science/article/pii/S0191886923002301). Conventionally, item desirability is rated by human respondents, who judge the degree to which endorsing an item would be perceived favorably or unfavorably by societal standards.

    This is closely linked to sentiment (i.e., the valence of an item). However, there are cases where sentiment and desirability diverge. The item "*I love a good fight*" has positive valence, but it's not particularly desirable.

    This is an excellent opportunity to demonstrate transfer learning. Transformer models have an excellent track record in accurately classifying sentiment. We'll use [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) as a base model and demonstrate how to train it to predict item desirability.

    As you may recall, classifier models output class membership probabilities. This is unfortunate, since we want our desirability prediction model to output one continuous score. We'll have to make some adaptations to the model architecture.

    But first things first: Let's load the required packages and some sample data from [Hughes et al. (2021)](https://doi.org/10.1027/1866-5888/a000267) which was used in [Hommel (2023)](https://www.sciencedirect.com/science/article/pii/S0191886923002301).
    """
    )
    return


@app.cell
def _(mo):
    import copy
    import torch
    import pandas as pd
    import numpy as np
    import torch.nn as nn
    import evaluate
    from scipy.stats import pearsonr

    from datasets import Dataset, load_dataset
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import DataCollatorWithPadding, TrainingArguments, Trainer, TrainerCallback

    mo.show_code()
    return (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Dataset,
        Trainer,
        TrainingArguments,
        copy,
        load_dataset,
        mean_absolute_error,
        mean_squared_error,
        nn,
        np,
        pd,
        pearsonr,
        torch,
    )


@app.cell
def _(mo, pd):
    desirability_data = pd.read_parquet("data/desirability.parquet").sort_values(by="desirability", ascending=False)
    mo.show_code()
    return (desirability_data,)


@app.cell
def _(desirability_data):
    desirability_data
    return


@app.cell
def _(mo):
    mo.md(r"""Look at the data and confirm that it makes intuitive sense. Low scores in the `desirability` column obviously reflect undesirable traits, if the corresponding item is endorsed. We'll use this as training data for domain adaptation later.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Base Model

    Next, we load the base model, namely [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) from the hugging face model hub. We'll store the tokenizer and the model into an object we call `base`.
    """
    )
    return


@app.cell
def _(AutoModelForSequenceClassification, AutoTokenizer, mo, torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base_model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    base = {
        'tokenizer': AutoTokenizer.from_pretrained(base_model_path),
        'model': AutoModelForSequenceClassification.from_pretrained(base_model_path).to(device)
    }
    mo.show_code()
    return base, device


@app.cell
def _(mo):
    mo.md(r"""As usual, let's inspect the model architecture.""")
    return


@app.cell
def _(base):
    base["model"]
    return


@app.cell
def _(mo):
    mo.md(r"""Now, let us write a simple prediction function, which we'll re-use at multiple occasions throughout this notebook.""")
    return


@app.cell
def _(mo, torch):
    def predict(text, model, tokenizer):
        device = next(model.parameters()).device

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        inputs = {key: value.to(device) for key, value in inputs.items()}

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = outputs.logits.squeeze()

        return prediction

    mo.show_code()
    return (predict,)


@app.cell
def _(mo):
    mo.md(r"""We'll test the prediction with three simple sentences about our attitudes towards different kinds of seafood, which we happen to have very strong opinions on 🐟🦞""")
    return


@app.cell
def _(base, mo, predict):
    base_model_test_prediction = predict(
        text=["i hate salmon", "I dislike tuna", "I like lobsters"], 
        model=base["model"], 
        tokenizer=base["tokenizer"]
    )

    mo.show_code()
    return (base_model_test_prediction,)


@app.cell
def _(base_model_test_prediction):
    base_model_test_prediction
    return


@app.cell
def _(mo):
    mo.md(r"""Above, we see the logits for our three statements, for the three categories negative, neutral, and positive. We should apply softmax to convert to probabilities, but we confirm that the prediction is intuitive by looking at the ranks.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Regressive Sentiment Model

    Before we can train this model on our item desirability data (which is continuous), we need to make some changes to the model architecture. Specifically, we want to frame the task as a regression task.

    We'll copy the base model to the `regressive_sentiment` object and set the projection layer to size 1, instead of 3.
    """
    )
    return


@app.cell
def _(base, copy, device, mo, nn):
    regressive_sentiment = copy.deepcopy(base)
    regressive_sentiment["model"].classifier.out_proj = nn.Linear(768, 1).to(device)
    regressive_sentiment["model"].num_labels = 1
    regressive_sentiment["model"].config.problem_type = "regression"

    mo.show_code()
    return (regressive_sentiment,)


@app.cell
def _(mo):
    mo.md(r"""Let's examine the architecture for the modified model, to confirm that the changes were applied.""")
    return


@app.cell
def _(regressive_sentiment):
    regressive_sentiment["model"]
    return


@app.cell
def _(mo):
    mo.md(r"""Great, but we're not done yet. We will need to re-train the model with sentiment data, to re-connect the model body with the model head. Before we do that, we want to make sure not to update the parameters in the model body, as this is where the bulk of knowledge in our model resides. We can do this by freezing layers, using the `param.requires_grad` flag. The model head should of course be able to update its parameters.""")
    return


@app.cell
def _(mo, regressive_sentiment):
    for param in regressive_sentiment["model"].parameters():
        param.requires_grad = False

    for param in regressive_sentiment["model"].classifier.parameters():    
        param.requires_grad = True

    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(r"""Now that we have successfully performed model surgery, let's again use the `predict()` function on our three pescatarian sentiments.""")
    return


@app.cell
def _(mo, predict, regressive_sentiment):
    untrained_regressive_sentiment_test_prediction = predict(
        text=[
            "i hate salmon", "I dislike tuna", "I like lobsters"
        ], 
        model=regressive_sentiment["model"], 
        tokenizer=regressive_sentiment["tokenizer"]
    )

    mo.show_code()
    return (untrained_regressive_sentiment_test_prediction,)


@app.cell
def _(untrained_regressive_sentiment_test_prediction):
    print(untrained_regressive_sentiment_test_prediction)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    We confirm that the shape of the model output is correct: We only get three values for three sentences!

    Unfortunately, they don't seem to make sense, as the positive sentence sits right in the middle between the two negative sentences. This shouldn't surprise us. The model was trained to predict three classes, and now only has one output neuron. We'll first have to train it to teach it to accept and live with its mutilation.

    To do so, we will use the [tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) dataset from Cardiff University, which contains the original training data for our base model.
    """
    )
    return


@app.cell
def _(load_dataset, mo):

    discrete_sentiment_data_path = "cardiffnlp/tweet_eval"
    discrete_sentiment_data_name = "sentiment"

    with mo.status.spinner(f"Loading {discrete_sentiment_data_path}"):
        discrete_sentiment_data = load_dataset(
            path=discrete_sentiment_data_path, 
            name=discrete_sentiment_data_name, 
            split="train[:1000]"
        ).to_pandas()

    mo.show_code()
    return discrete_sentiment_data, discrete_sentiment_data_path


@app.cell
def _(discrete_sentiment_data):
    discrete_sentiment_data
    return


@app.cell
def _(mo):
    mo.md(r"""This data is categorically labeled, which poses a problem for training our now regressive sentiment model. We will have to re-label the data. To do so, we'll first pass it to the predict function and obtain softmax probabilities for each class.""")
    return


@app.cell
def _(
    base,
    discrete_sentiment_data,
    discrete_sentiment_data_path,
    mo,
    predict,
):
    with mo.status.spinner(f"Predicting class membership for {discrete_sentiment_data_path} data"):
        discrete_sentiment_data_predictions = predict(
            text=discrete_sentiment_data['text'].tolist(), 
            model=base["model"], 
            tokenizer=base["tokenizer"]
        )
    mo.show_code()
    return (discrete_sentiment_data_predictions,)


@app.cell
def _(discrete_sentiment_data_predictions, mo, pd, torch):
    metric_sentiment_data = pd.DataFrame(
        data=torch.softmax(discrete_sentiment_data_predictions, axis=1).cpu(), 
        columns=['negative', 'neutral', 'positive']
    )

    mo.show_code()
    return (metric_sentiment_data,)


@app.cell
def _(mo):
    mo.md(r"""Next, let's subtract the negative class membership probability from the positive class membership probability to get a crude continuous sentiment score, which we'll call `labels`.""")
    return


@app.cell
def _(discrete_sentiment_data, metric_sentiment_data, mo):
    metric_sentiment_data['labels'] = metric_sentiment_data['positive'] - metric_sentiment_data['negative']
    metric_sentiment_data['text'] = discrete_sentiment_data['text'].tolist()
    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(r"""Now it's time to prepare the data for model training. We'll use a data collator which automatically performs some batching and padding of the training data. We then tokenize our text column.""")
    return


@app.cell
def _(metric_sentiment_data):
    metric_sentiment_data
    return


@app.cell
def _(
    DataCollatorWithPadding,
    Dataset,
    metric_sentiment_data,
    regressive_sentiment,
):
    data_collator = DataCollatorWithPadding(tokenizer=regressive_sentiment["tokenizer"])

    def sentiment_tokenize_function(examples):
        tokenized = regressive_sentiment["tokenizer"](
            examples['text'], 
            truncation=True, 
            padding=False,
            max_length=512
        )
        return tokenized

    metric_sentiment_training_data = Dataset.from_pandas(metric_sentiment_data).map(sentiment_tokenize_function, batched=True, num_proc=1)
    return data_collator, metric_sentiment_training_data


@app.cell
def _(metric_sentiment_training_data):
    metric_sentiment_training_data
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Now we will have to decide on a training configuration and hyperparameters. There is a lot to be said here, but for this demonstration, we'll only use a minimal configuration. You can read more about training arguments in the [hugging face docs](https://huggingface.co/docs/transformers/en/main_classes/trainer).

    A few important choices include:

    - `max_steps` / `num_train_epochs`: One training step is one batch in the training data; one epoch is the entire dataset.
    - `learning_rate`: The size of steps when updating the model parameters. Higher learning rates lead to faster convergence and help escape small local minima, but can cause instability or overstep the optimal point of convergence.
    - `weight_decay`: A penalty term for regularization (preventing overfit), by encouraging the model to adjust weights conservatively.
    - `batch_size`/ `per_device_train_batch_size`: How many training examples get passed in one training step. Smaller batch sizes may yield better generalization, while larger batch sizes show more stable gradient updates.
    """
    )
    return


@app.cell
def _(TrainingArguments, mo):
    training_args = TrainingArguments(
        output_dir='../output/',
        #max_steps=10,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="no",
        logging_dir='../output/',
        logging_steps=1,
        report_to=[],
        disable_tqdm=False,
    )

    mo.show_code()
    return (training_args,)


@app.cell
def _(mo):
    mo.md(r"""Now that we have all the pieces in place, we can finally put together our training function, which will require the model, the training arguments, the training data, and the data collator.""")
    return


@app.cell
def _(
    Trainer,
    copy,
    mo,
    set_regressive_sentiment_history,
    set_trained_regressive_sentiment,
):
    async def train_regressive_sentiment(model, training_args, data_collator, train_dataset):
        tmp = copy.deepcopy(model)

        trainer = Trainer(
            model=tmp["model"],
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )

        training_output  = trainer.train()

        set_trained_regressive_sentiment(tmp)
        set_regressive_sentiment_history(trainer.state.log_history)
        mo.output.append(
            list(training_output)
        )

    mo.show_code()
    return (train_regressive_sentiment,)


@app.cell
def _(mo):
    train_regressive_sentiment_button = mo.ui.run_button(
        label="Train model for regressive sentiment prediction",
        full_width=True
    )
    mo.vstack([
        mo.md("Press the button to train the model and hope for the best!"),
        train_regressive_sentiment_button    
    ])
    return (train_regressive_sentiment_button,)


@app.cell
def _(get_regressive_sentiment_history, get_trained_regressive_sentiment, mo):
    mo.stop(not get_trained_regressive_sentiment())
    mo.md(f"""
    Below, you'll see the loss for each of the {len(get_regressive_sentiment_history())} training steps:
    """)
    return


@app.cell
async def _(
    data_collator,
    metric_sentiment_training_data,
    mo,
    regressive_sentiment,
    train_regressive_sentiment,
    train_regressive_sentiment_button,
    training_args,
):
    if train_regressive_sentiment_button.value:

        with mo.status.spinner("Training..."):
            await train_regressive_sentiment(
                model=regressive_sentiment,
                training_args=training_args,
                data_collator=data_collator,
                train_dataset=metric_sentiment_training_data
            )

    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(r"""We plot the loss over training steps to see if we have made some progress.""")
    return


@app.cell
def _(
    get_regressive_sentiment_history,
    get_trained_regressive_sentiment,
    mo,
    pd,
):
    mo.stop(not get_trained_regressive_sentiment())
    pd.DataFrame(get_regressive_sentiment_history()).plot(y="loss")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Great, the curve indicates that the model was able to learn and minimize the prediction error. Ideally, we should continue training until convergence has stabilized, but we'll leave it at this for now. 

    Let us use the trained regressive sentiment model to again predict the sentiment of our opinions about fish.
    """
    )
    return


@app.cell
def _(get_trained_regressive_sentiment, mo, predict):
    mo.stop(not get_trained_regressive_sentiment(), "Train regressive sentiment model to continue")

    mo.output.append(
        predict(
            text=["i hate salmon", "I dislike tuna", "I like lobsters"] , 
            model=get_trained_regressive_sentiment()["model"], 
            tokenizer=get_trained_regressive_sentiment()["tokenizer"]
        )
    )
    return


@app.cell
def _(mo):
    mo.md(r"""Wonderful, these predictions make much more sense! We can now go on with training our final model for item desirability prediction.""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Desirability Model

    Now that we have a regressive sentiment model, we can employ domain adaptation (i.e., teaching the model a different content domain, without architectural changes).

    First, we split our item desirability data into three partitions.
    """
    )
    return


@app.cell
def _(Dataset, desirability_data, mo):
    dataset = Dataset.from_pandas(desirability_data.rename(columns={'desirability': 'labels'}))

    seed = 42
    train_test = dataset.train_test_split(test_size=0.3, seed=seed)
    test_eval = train_test['test'].train_test_split(test_size=0.5, seed=seed)

    mo.show_code()
    return test_eval, train_test


@app.cell
def _(mo, test_eval, train_test):
    mo.output.append(
        mo.vstack([
            train_test,
            test_eval
        ])
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    The majority of the data (`train`) is used to train the model. The `dev`/`eval` set is used during training, to monitor the model's performance and adjust hyperparameters. The `test` set is only used after model training is complete and constitutes our final evaluation.

    Next, we again prepare the data by tokenizing the text.
    """
    )
    return


@app.cell
def _(mo, regressive_sentiment, test_eval, train_test):
    def desirability_tokenize_function(examples):
        tokenized = regressive_sentiment["tokenizer"](
            examples['text'], 
            truncation=True, 
            padding=False,
            max_length=512
        )

        return tokenized

    train_dataset = train_test['train'].map(desirability_tokenize_function, batched=True, num_proc=1)
    test_dataset = test_eval['test'].map(desirability_tokenize_function, batched=True, num_proc=1)
    eval_dataset = test_eval['train'].map(desirability_tokenize_function, batched=True, num_proc=1)

    mo.show_code()
    return eval_dataset, test_dataset, train_dataset


@app.cell
def _(test_dataset):
    test_dataset
    return


@app.cell
def _(mo):
    mo.md(r"""To evaluate the training process, let us write a small function that computes some performance metrics, such as the mean absolute error, mean squared error, and the correlation between the model predictions and the true labels.""")
    return


@app.cell
def _(mean_absolute_error, mean_squared_error, mo, np, pearsonr):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        labels = labels.flatten()

        mae = mean_absolute_error(labels, predictions)
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        correlation, _ = pearsonr(labels, predictions)

        return {
            "mae": mae,
            "mse": mse, 
            "rmse": rmse,
            "correlation": correlation
        }

    mo.show_code()
    return (compute_metrics,)


@app.cell
def _(mo):
    mo.md(r"""Now we set the training arguments for domain adaptation. Since we're only using a small subset of the data used in [Hommel (2023)](https://www.sciencedirect.com/science/article/pii/S0191886923002301), we chose a smaller batch size and learning rate. We also increase the weight decay.""")
    return


@app.cell
def _(TrainingArguments, mo):
    desirability_training_args = TrainingArguments(
        output_dir='../output/',
        num_train_epochs=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-6,
        weight_decay=0.05,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        save_steps=50,
        load_best_model_at_end=False,
        metric_for_best_model="correlation",
        greater_is_better=True,
        logging_dir='../output/',
        logging_steps=1,
        report_to=[],
        disable_tqdm=False,
        warmup_steps=20,
        save_total_limit=2,
    )

    mo.show_code()
    return (desirability_training_args,)


@app.cell
def _(mo):
    mo.md(r"""Again, we define a training function that puts everything together.""")
    return


@app.cell
def _(
    Trainer,
    copy,
    mo,
    set_regressive_desirability_history,
    set_trained_regressive_desirability,
):
    async def train_desirability_model(model, training_args, data_collator, train_dataset, eval_dataset, compute_metrics):
        tmp = copy.deepcopy(model)

        for param in tmp["model"].parameters():
            param.requires_grad = True

        trainer = Trainer(
            model=tmp["model"],
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        training_output  = trainer.train()

        set_trained_regressive_desirability(tmp)
        set_regressive_desirability_history(trainer.state.log_history)

        mo.output.append(
            list(training_output)
        )

    mo.show_code()
    return (train_desirability_model,)


@app.cell
def _(mo):
    train_regressive_desirability_button = mo.ui.run_button(
        label="Train model for regressive desirability prediction",
        full_width=True
    )
    mo.vstack([
        mo.md("Press the button to train the model and hope for the best!"),
        train_regressive_desirability_button    
    ])
    return (train_regressive_desirability_button,)


@app.cell
async def _(
    compute_metrics,
    data_collator,
    desirability_training_args,
    eval_dataset,
    mo,
    regressive_sentiment,
    train_dataset,
    train_desirability_model,
    train_regressive_desirability_button,
):
    if train_regressive_desirability_button.value:

        with mo.status.spinner("Training..."):
            await train_desirability_model(
                model=regressive_sentiment,
                training_args=desirability_training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics
            )

    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(r"""The logfile shows that errors decrease and correlation between the predictions and the labels increase over time. Nice! Nevertheless, let's look at the plot:""")
    return


@app.cell
def _(
    get_regressive_desirability_history,
    get_trained_regressive_desirability,
    mo,
    pd,
):
    mo.stop(not get_trained_regressive_desirability())
    pd.DataFrame(get_regressive_desirability_history()).plot(y="loss")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    What is happening here? The training process looks extremely volatile. While the loss is decreasing overall, the enormous spikes are problematic because they make training unreliable and unpredictable. We can't trust the process to work consistently or know when we have achieved good results. With extremely little training data, this volatility is often expected.

    We could optimize hyperparameters, ideally with a framework like [optuna](https://github.com/optuna/optuna), but in our case, we can be pretty sure that we would need more training data for stable results. We'll have to accept this unstable model for now.

    Finally, we'll define a small evaluation function, to be able to compare results between the sentiment and the desirability model.
    """
    )
    return


@app.cell
def _(mean_absolute_error, mean_squared_error, mo, np, pearsonr, predict):
    def evaluate_model(model, tokenizer, dataset):

        text_list = list(dataset['text'])

        predictions = predict(
            text=text_list, 
            model=model, 
            tokenizer=tokenizer
        )

        predictions = predictions.cpu().numpy().flatten()
        labels = np.array(dataset['labels'])

        mae = mean_absolute_error(labels, predictions)
        mse = mean_squared_error(labels, predictions)
        rmse = np.sqrt(mse)
        correlation, _ = pearsonr(labels, predictions)

        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "correlation": correlation
        }

    mo.show_code()
    return (evaluate_model,)


@app.cell
def _(mo):
    mo.md(r"""At last, let us compare the models and also print the difference between the `test` and `eval` set to get an appreciation of potential issues with overfitting to the data.""")
    return


@app.cell
def _(
    eval_dataset,
    evaluate_model,
    get_trained_regressive_desirability,
    get_trained_regressive_sentiment,
    mo,
    pd,
    test_dataset,
):
    mo.stop(not get_trained_regressive_sentiment(), "Train regressive sentiment model to continue")

    if get_trained_regressive_desirability():

        metrics_eval_desirability = evaluate_model(
            model=get_trained_regressive_desirability()['model'], 
            tokenizer=get_trained_regressive_desirability()['tokenizer'],
            dataset=eval_dataset
        )

        metrics_test_desirability = evaluate_model(
            model=get_trained_regressive_desirability()['model'], 
            tokenizer=get_trained_regressive_desirability()['tokenizer'],
            dataset=test_dataset
        )

        metrics_eval_sentiment = evaluate_model(
            model=get_trained_regressive_sentiment()['model'], 
            tokenizer=get_trained_regressive_sentiment()['tokenizer'],
            dataset=test_dataset
        )

        eval_df = pd.DataFrame(
            data=[metrics_eval_desirability, metrics_test_desirability, metrics_eval_sentiment], 
            index=[
                "Desirability Model (eval)",
                "Desirability Model (test)",
                "Sentiment Model (eval)",
            ]
        )

    mo.show_code()
    return (eval_df,)


@app.cell
def _(eval_df, get_trained_regressive_sentiment, mo):
    mo.stop(not get_trained_regressive_sentiment(), "Train regressive sentiment model to continue")
    mo.show_code(
        mo.ui.table(eval_df)
    )
    return


if __name__ == "__main__":
    app.run()
