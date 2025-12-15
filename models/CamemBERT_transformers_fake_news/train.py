import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from datasets import Dataset

from transformers import (
    CamembertTokenizer,
    CamembertForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import accuracy_score, f1_score


df = pd.read_csv("hf://datasets/readerbench/fakenews-climate-fr/fake-fr.csv")


label_encoder = LabelEncoder()
df["Label"] = label_encoder.fit_transform(df["Label"])
print("Mapping labels :", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Label"])
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


tokenizer = CamembertTokenizer.from_pretrained("camembert-base")

def tokenize(batch):
    return tokenizer(
        batch["Text"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)


train_dataset = train_dataset.rename_column("Label", "labels")  # torch / HF , accepte labels et non "Label"
test_dataset = test_dataset.rename_column("Label", "labels")

cols_to_keep = ["input_ids", "attention_mask", "labels"]
train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in cols_to_keep])
test_dataset = test_dataset.remove_columns([c for c in test_dataset.column_names if c not in cols_to_keep])

train_dataset.set_format("torch")
test_dataset.set_format("torch")


class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df["Label"]),
    y=df["Label"]
)
print("Poids de classes =", class_weights)


model = CamembertForSequenceClassification.from_pretrained(
    "camembert-base",
    num_labels=len(label_encoder.classes_)
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro")
    }


    training_args = TrainingArguments(
    output_dir="/teamspace/studios/this_studio/CamemBERT_fake_news/results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="/teamspace/studios/this_studio/CamemBERT_fake_news/logs",
    weight_decay=0.01,
    learning_rate=5e-5,
)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.class_weights = (
            torch.tensor(class_weights, dtype=torch.float)
            if not isinstance(class_weights, torch.Tensor)
            else class_weights.clone().detach()
            if class_weights is not None
            else None
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
            if self.class_weights is not None else None
        )

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    class_weights=class_weights
)


batch = next(iter(train_dataset))
print("input_ids shape:", batch["input_ids"].shape)  # doit être [batch_size, seq_len]


trainer.train()