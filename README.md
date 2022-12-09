# Baselines for Metaphor Detection

This repository contains a very simple baseline implementation for metaphor detection.
It is designed to give students working with metaphor detection a quick and easy entry point to start their own experiments.


## Requirements

```
transformers==4.18.0
datasets==2.2.2
sklearn==1.1.3
```

## Downloading Data

To download the VUA corpus, simply run ``python -m metabaseline.prepare_data`` from the repository root. This will download the raw files, convert the token level annotations into character level annotations and save them as in a jsonl format. Each entry is one sentence, ``instance`` contains all annotated tokens in the sentence, where 1 indicates metaphoricity and 0 indicates literal instances.

```
{
    {
        "txt_id": "<text id>",
        "sentence_id": "<sentence id>",
        "sentence": "<sentence text>",
        "instances": [
            {
                "char_range": [<first char idx>, <last char idx>],
                "label": <0/1 label>}
            },
            ...
        }
}
```


## Training

Currently the repository only supports training a single Huggingface model for Sequence classification using the standard trainer API. You can find the relevant code in ``metabaseline.train_seq_label``

``bert-base-uncased`` achieves an F1 score of ``0.70`` (Accuracy: ``0.90``) on VUA-ALL-POS, which is in line with prior reported baseline results (https://aclanthology.org/2020.coling-main.332.pdf, Table 2).

## Code Structure

All code for downloading and saving the datasets can be found in ``metabaseline.prepare_data``. Training code is in ``metabaseline.train_seq_label``.
If you work within the transformers ecosystem, you can use the function ``metabaseline.data.load_and_process_dataset`` to directly load a ``Dataset`` instance that contains tokenized sentences and corresponding token-level labels.
