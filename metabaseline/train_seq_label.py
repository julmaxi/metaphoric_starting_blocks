import transformers
import numpy as np
from sklearn.metrics import f1_score

from .data import load_and_process_dataset


def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = load_and_process_dataset("data/all_pos_train", tokenizer=tokenizer)
    test_dataset = load_and_process_dataset("data/all_pos_test", tokenizer=tokenizer)

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
        save_steps=1000,
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="epoch",
        eval_delay=100,

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
        eval_dataset=None,
        compute_metrics=compute_metrics,

        #data_collator=collator,
    )

    trainer.train()

    results = trainer.predict(test_dataset)
    print(results.metrics)

if __name__ == "__main__":
    main()
