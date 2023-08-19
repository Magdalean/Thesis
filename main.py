import numpy as np
import tensorflow as tf
import pysbd
import requests
import pandas as pd
import os
import random

random.seed(97)

from transformers import (
    AutoTokenizer,
)
from tensorflow import keras
from datasets import Dataset
from datasets import DatasetDict
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoTokenizer
from transformers import TFBertForNextSentencePrediction
import tensorflow_privacy
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy


def create_dataset(loaded_CNN, bagdf):
    text = loaded_CNN["article"].values.tolist()
    bag_list = bagdf.values.tolist()
    bag_clean = [str(x) for x in bag_list]
    bag = [x[2 : len(x) - 2] for x in bag_clean]
    bag_size = len(bag)

    sentence_a = []
    sentence_b = []
    label = []

    for paragraph in text:
        sentences = [sentence for sentence in paragraph.split(".") if sentence != ""]
        num_sentences = len(sentences)
        # not adding single sentences to the data
        if num_sentences > 1:
            start = random.randint(
                0, num_sentences - 2
            )  # -2 because we want to have a sentence following the start sentence
            sentence_a.append(sentences[start])
            if random.random() > 0.5:
                sentence_b.append(bag[random.randint(0, bag_size - 1)])
                label.append(1)  # 1 when it is not the next sentence
            else:
                sentence_b.append(sentences[start + 1])
                label.append(0)  # 0 when it is the next sentence

    df_dataset = pd.DataFrame(
        {"lst1Title": sentence_a, "lst2Title": sentence_b, "label": label}
    )
    train_dataset = Dataset.from_pandas(df_dataset).train_test_split(test_size=0.25)
    test_dataset = train_dataset["test"].train_test_split(test_size=0.5, seed=42)

    return_dataset = DatasetDict(
        {
            "train": train_dataset["train"],
            "valid": test_dataset["train"],
            "test": test_dataset["test"],
        }
    )
    return return_dataset


def tokenize_dataset(tokenizer, dataset):
    tokenized_data = tokenizer(
        dataset["lst1Title"],
        dataset["lst2Title"],
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=512,
    )
    tokenized_data = dict(tokenized_data)
    labels = np.array(dataset["label"])

    return (tokenized_data, labels)


def training(model_name, model, X_train, y_train, X_valid, y_valid, X_test):
    EPOCHS = 2
    BATCH_SIZE = 2  # changed from 28
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.000005)
    model.compile(optimizer=optimizer, loss=loss, metrics="accuracy")
    model_history = model.fit(
        (X_train),
        (y_train),
        validation_data=(X_valid, y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    y_pred = model.predict(X_test)
    root_path = "Models/"
    os.makedirs(os.path.dirname(root_path), exist_ok=True)
    model.save_pretrained(root_path + model_name)
    print("Model saved")

    return (model, y_pred)


def training_priv(
    noise_multiplier,
    model_name,
    model_priv,
    X_train,
    y_train,
    X_valid,
    y_valid,
    X_test,
    y_test,
):
    EPOCHS = 2
    BATCH_SIZE = 2  # changed from 28
    l2_norm_clip = 1
    num_microbatches = 1
    learning_rate = 0.000005  # changed from 0.25
    # loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    print("For noise multiplier = ", noise_multiplier)

    # optimizer_priv = tensorflow_privacy.DPKerasSGDOptimizer(
    # l2_norm_clip=l2_norm_clip,
    # noise_multiplier=noise_multiplier,
    # num_microbatches=num_microbatches,
    # learning_rate=learning_rate)

    optimizer_priv = tensorflow_privacy.DPKerasAdamOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate,
    )

    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE
    )

    print(
        compute_dp_sgd_privacy.compute_dp_sgd_privacy_statement(
            number_of_examples=X_train["input_ids"].shape[0],
            batch_size=BATCH_SIZE,
            noise_multiplier=noise_multiplier,
            num_epochs=EPOCHS,
            delta=1e-5,
        )
    )
    model_priv.compile(optimizer=optimizer_priv, loss=loss, metrics="accuracy")
    model_history = model_priv.fit(
        (X_train),
        (y_train),
        validation_data=(X_valid, y_valid),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    y_pred = model_priv.predict(X_test)

    root_path = "Models/"
    model_priv.save_pretrained(root_path + model_name + "_" + str(noise_multiplier))
    print("Private model saved")
    [Accuracy, Precision, Recall, F1, AUPRC, AUROC] = metric_calc(y_pred, y_test)
    print(
        "Accuracy: ",
        Accuracy,
        "Precision: ",
        Precision,
        "Recall: ",
        Recall,
        "F1: ",
        F1,
        "AUPRC: ",
        AUPRC,
        "AUROC: ",
        AUROC,
    )

    return (Accuracy, Precision, Recall, F1, AUPRC, AUROC)


def metric_calc(pred, test):
    y_pred_reformat = np.zeros((len(pred[0])))
    for i in range(len(pred[0])):
        y_pred_reformat[i] = np.argmax(pred[0][i])
    y_pred_reformat = pd.get_dummies(y_pred_reformat).values

    test_flat = np.argmax(test, axis=1)
    pred_flat = np.argmax(y_pred_reformat, axis=1)

    Accuracy = accuracy_score(test_flat, pred_flat)
    Precision = round(precision_score(test_flat, pred_flat, average="binary"), 4)
    Recall = round(recall_score(test_flat, pred_flat, average="binary"), 4)
    F1 = round(f1_score(test_flat, pred_flat, average="binary"), 4)

    # AUPRC and AUROC
    probabilities = tf.nn.softmax(pred.logits, axis=None, name=None)
    AUROC = round(roc_auc_score(test_flat, probabilities[:, 1]), 4)
    AUPRC = round(average_precision_score(test_flat, probabilities[:, 1]), 4)

    return (Accuracy, Precision, Recall, F1, AUPRC, AUROC)


def one_hot(arr):
    res = np.zeros((arr.size, arr.max() + 1))
    res[np.arange(arr.size), arr] = 1
    return res


seg = pysbd.Segmenter(language="en", clean=False)

loaded_CNN1 = pd.read_csv("data/CNN_20K_1.csv")
loaded_CNN2 = pd.read_csv("data/CNN_20K_2.csv")

bagdf1 = pd.read_csv("data/bag20k.csv", index_col=0)
bagdf2 = pd.read_csv("data/bag20k-2.csv", index_col=0)
CNN_dataset = create_dataset(loaded_CNN1, bagdf1)
CNN_dataset2 = create_dataset(loaded_CNN2, bagdf2)


def training_BERT(
    MODEL_NAME, label, X_train, y_train, X_valid, y_valid, X_test, y_test
):
    acc = []
    pre = []
    rec = []
    f1 = []
    auroc = []
    auprc = []

    if MODEL_NAME == "medicalai/ClinicalBERT":
        model = TFBertForNextSentencePrediction.from_pretrained(
            MODEL_NAME, from_pt=True
        )
    else:
        model = TFBertForNextSentencePrediction.from_pretrained(MODEL_NAME)
    [finetuned_model, y_pred] = training(
        label, model, X_train, y_train, X_valid, y_valid, X_test
    )

    [Accuracy, Precision, Recall, F1, AUPRC, AUROC] = metric_calc(y_pred, y_test)
    print(Accuracy, Precision, Recall, F1, AUPRC, AUROC)

    acc.append(Accuracy)
    pre.append(Precision)
    rec.append(Recall)
    f1.append(F1)
    auprc.append(AUPRC)
    auroc.append(AUROC)

    # Private
    for noise_multiplier in [1.0791, 2.6945, 4.224, 21.7, 51, 415]:
        if MODEL_NAME == "medicalai/ClinicalBERT":
            model_private = TFBertForNextSentencePrediction.from_pretrained(
                MODEL_NAME, from_pt=True
            )
        else:
            model_private = TFBertForNextSentencePrediction.from_pretrained(MODEL_NAME)

        [
            Accuracy_priv,
            Precision_priv,
            Recall_priv,
            F1_priv,
            AUPRC_priv,
            AUROC_priv,
        ] = training_priv(
            noise_multiplier,
            label,
            model_private,
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
        )
        acc.append(Accuracy_priv)
        pre.append(Precision_priv)
        rec.append(Recall_priv)
        f1.append(F1_priv)
        auprc.append(AUPRC_priv)
        auroc.append(AUROC_priv)

    return (acc, pre, rec, f1, auprc, auroc)


# BERT BASE
MODEL_NAME = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

X_train_bert, y_train_bert = tokenize_dataset(tokenizer, CNN_dataset["train"])
X_valid_bert, y_valid_bert = tokenize_dataset(tokenizer, CNN_dataset["valid"])
X_test_bert, y_test_bert = tokenize_dataset(tokenizer, CNN_dataset["test"])

y_train_bert = one_hot(y_train_bert)
y_valid_bert = one_hot(y_valid_bert)
y_test_bert = one_hot(y_test_bert)

X_train_bert2, y_train_bert2 = tokenize_dataset(tokenizer, CNN_dataset2["train"])
X_valid_bert2, y_valid_bert2 = tokenize_dataset(tokenizer, CNN_dataset2["valid"])
X_test_bert2, y_test_bert2 = tokenize_dataset(tokenizer, CNN_dataset2["test"])

y_train_bert2 = one_hot(y_train_bert2)
y_valid_bert2 = one_hot(y_valid_bert2)
y_test_bert2 = one_hot(y_test_bert2)


MODEL_NAME = "medicalai/ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

X_train_cbert, y_train_cbert = tokenize_dataset(tokenizer, CNN_dataset["train"])
X_valid_cbert, y_valid_cbert = tokenize_dataset(tokenizer, CNN_dataset["valid"])
X_test_cbert, y_test_cbert = tokenize_dataset(tokenizer, CNN_dataset["test"])

y_train_cbert = one_hot(y_train_cbert)
y_valid_cbert = one_hot(y_valid_cbert)
y_test_cbert = one_hot(y_test_cbert)

# CLINICAL BERT
MODEL_NAME = "medicalai/ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

X_train_cbert, y_train_cbert = tokenize_dataset(tokenizer, CNN_dataset["train"])
X_valid_cbert, y_valid_cbert = tokenize_dataset(tokenizer, CNN_dataset["valid"])
X_test_cbert, y_test_cbert = tokenize_dataset(tokenizer, CNN_dataset["test"])

y_train_cbert = one_hot(y_train_cbert)
y_valid_cbert = one_hot(y_valid_cbert)
y_test_cbert = one_hot(y_test_cbert)


X_train_cbert2, y_train_cbert2 = tokenize_dataset(tokenizer, CNN_dataset2["train"])
X_valid_cbert2, y_valid_cbert2 = tokenize_dataset(tokenizer, CNN_dataset2["valid"])
X_test_cbert2, y_test_cbert2 = tokenize_dataset(tokenizer, CNN_dataset2["test"])

y_train_cbert2 = one_hot(y_train_cbert2)
y_valid_cbert2 = one_hot(y_valid_cbert2)
y_test_cbert2 = one_hot(y_test_cbert2)


# BIO BERRT ---- might not be needed

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

X_train_bc, y_train_bc = tokenize_dataset(tokenizer, CNN_dataset["train"])
X_valid_bc, y_valid_bc = tokenize_dataset(tokenizer, CNN_dataset["valid"])
X_test_bc, y_test_bc = tokenize_dataset(tokenizer, CNN_dataset["test"])

y_train_bc = one_hot(y_train_bc)
y_valid_bc = one_hot(y_valid_bc)
y_test_bc = one_hot(y_test_bc)

X_train_bc2, y_train_bc2 = tokenize_dataset(tokenizer, CNN_dataset2["train"])
X_valid_bc2, y_valid_bc2 = tokenize_dataset(tokenizer, CNN_dataset2["valid"])
X_test_bc2, y_test_bc2 = tokenize_dataset(tokenizer, CNN_dataset2["test"])

y_train_bc2 = one_hot(y_train_bc2)
y_valid_bc2 = one_hot(y_valid_bc2)
y_test_bc2 = one_hot(y_test_bc2)

# BERT BASE
acc_BERT, pre_BERT, rec_BERT, f1_BERT, auprc_BERT, auroc_BERT = training_BERT(
    "bert-base-cased",
    "BERT",
    X_train_bert,
    y_train_bert,
    X_valid_bert,
    y_valid_bert,
    X_test_bert,
    y_test_bert,
)

(
    acc_BERTv2,
    pre_BERTv2,
    rec_BERTv2,
    f1_BERTv2,
    auprc_BERTv2,
    auroc_BERTv2,
) = training_BERT(
    "bert-base-cased",
    "BERTv2",
    X_train_bert2,
    y_train_bert2,
    X_valid_bert2,
    y_valid_bert2,
    X_test_bert2,
    y_test_bert2,
)

# CLINICAL BERT
acc_cBERT, pre_cBERT, rec_cBERT, f1_cBERT, auprc_cBERT, auroc_cBERT = training_BERT(
    "bert-base-cased",
    "ClinicalBERT",
    X_train_cbert,
    y_train_cbert,
    X_valid_cbert,
    y_valid_cbert,
    X_test_cbert,
    y_test_cbert,
)

(
    acc_cBERTv2,
    pre_cBERTv2,
    rec_cBERTv2,
    f1_cBERTv2,
    auprc_cBERTv2,
    auroc_cBERTv2,
) = training_BERT(
    "bert-base-cased",
    "ClinicalBERTv2",
    X_train_cbert2,
    y_train_cbert2,
    X_valid_cbert2,
    y_valid_cbert2,
    X_test_cbert2,
    y_test_cbert2,
)

# BIO BERT -- might not be needed

(
    acc_bcBERT,
    pre_bcBERT,
    rec_bcBERT,
    f1_bcBERT,
    auprc_bcBERT,
    auroc_bcBERT,
) = training_BERT(
    "bert-base-cased",
    "ClinicalBioBERT",
    X_train_bc,
    y_train_bc,
    X_valid_bc,
    y_valid_bc,
    X_test_bc,
    y_test_bc,
)

(
    acc_bcBERTv2,
    pre_bcBERTv2,
    rec_bcBERTv2,
    f1_bcBERTv2,
    auprc_bcBERTv2,
    auroc_bcBERTv2,
) = training_BERT(
    "bert-base-cased",
    "ClinicalBioBERTv2",
    X_train_bc2,
    y_train_bc2,
    X_valid_bc2,
    y_valid_bc2,
    X_test_bc2,
    y_test_bc2,
)


metrics = ["Accuracy", "Precision", "Recall", "F1", "AUPRC", "AUROC"]
Epsilon = ["Inf", "15", "5", "3", "0.5", "0.2", "0.02"]

results_path = "results"
os.makedirs(os.path.dirname(results_path), exist_ok=True)

metrics_df_BERT = pd.DataFrame(
    list(zip(acc_BERT, pre_BERT, rec_BERT, f1_BERT, auprc_BERT, auroc_BERT)),
    index=Epsilon,
    columns=metrics,
)
metrics_df_BERT.to_csv("results/metrics_df_BERT.csv")
metrics_df_BERTv2 = pd.DataFrame(
    list(
        zip(acc_BERTv2, pre_BERTv2, rec_BERTv2, f1_BERTv2, auprc_BERTv2, auroc_BERTv2)
    ),
    index=Epsilon,
    columns=metrics,
)
metrics_df_BERTv2.to_csv("results/metrics_df_BERTv2.csv")


metrics_df_cBERT = pd.DataFrame(
    list(zip(acc_cBERT, pre_cBERT, rec_cBERT, f1_cBERT, auprc_cBERT, auroc_cBERT)),
    index=Epsilon,
    columns=metrics,
)
metrics_df_cBERT.to_csv("results/metrics_df_cBERT.csv")
metrics_df_cBERTv2 = pd.DataFrame(
    list(
        zip(
            acc_cBERTv2,
            pre_cBERTv2,
            rec_cBERTv2,
            f1_cBERTv2,
            auprc_cBERTv2,
            auroc_cBERTv2,
        )
    ),
    index=Epsilon,
    columns=metrics,
)
metrics_df_cBERTv2.to_csv("results/metrics_df_cBERTv2.csv")

metrics_df_bcBERT = pd.DataFrame(
    list(
        zip(acc_bcBERT, pre_bcBERT, rec_bcBERT, f1_bcBERT, auprc_bcBERT, auroc_bcBERT)
    ),
    index=Epsilon,
    columns=metrics,
)
metrics_df_bcBERT.to_csv("results/metrics_df_bcBERT.csv")
metrics_df_bcBERTv2 = pd.DataFrame(
    list(
        zip(
            acc_bcBERTv2,
            pre_bcBERTv2,
            rec_bcBERTv2,
            f1_bcBERTv2,
            auprc_bcBERTv2,
            auroc_bcBERTv2,
        )
    ),
    index=Epsilon,
    columns=metrics,
)
metrics_df_bcBERTv2.to_csv("results/metrics_df_bcBERTv2.csv")

print(metrics_df_BERT)
print(metrics_df_BERTv2)
print(metrics_df_cBERT)
print(metrics_df_cBERTv2)
print(metrics_df_bcBERT)
print(metrics_df_bcBERTv2)
