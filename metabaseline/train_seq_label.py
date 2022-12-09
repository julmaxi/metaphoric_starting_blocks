import transformers
import numpy as np
from sklearn.metrics import f1_score

import argparse


from .data import load_and_process_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", dest="train_path", default="data/vua_all_pos_train.jsonl")
    parser.add_argument("-T", "--test", dest="test_path", default="data/vua_all_pos_test.jsonl")
    parser.add_argument("-d", "--dev", dest="dev_path", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    if args.dev_path is None:
        train_dev_dataset = load_and_process_dataset(args.train_path, tokenizer=tokenizer)
        train_dev_dataset = train_dev_dataset.train_test_split(0.1)
        train_dataset = train_dev_dataset["train"]
        dev_dataset = train_dev_dataset["test"]
    else:
        train_dataset = load_and_process_dataset(args.train_path, tokenizer=tokenizer)
        dev_dataset = load_and_process_dataset(args.dev_path, tokenizer=tokenizer)

    test_dataset = load_and_process_dataset(args.test_path, tokenizer=tokenizer)

    model = transformers.AutoModelForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )

    lr = 5e-5
    training_args = transformers.TrainingArguments(
        output_dir=f"met_train",
        num_train_epochs=3,
        learning_rate=lr,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=64,
        warmup_ratio=0.06,
        weight_decay=0.01,
        fp16=True,
        report_to="none",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="epoch",
    )

    def compute_metrics(eval_results):
        predicted = np.array(eval_results.predictions).argmax(axis=-1)
        gold = np.array(eval_results.label_ids)[:,0,:]

        predicted = predicted[gold != -100]
        gold = gold[gold != -100]

        return {"accuracy": (gold == predicted).mean(), "f1": f1_score(gold, predicted)}

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,

        #data_collator=collator,
    )

    trainer.train()

    results = trainer.predict(test_dataset)
    print(results.metrics)

if __name__ == "__main__":
    main()
