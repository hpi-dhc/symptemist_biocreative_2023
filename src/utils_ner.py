import pandas as pd
import datasets
from datasets import Dataset, load_dataset
import tqdm
import numpy as np
from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification, DataCollatorForTokenClassification, AutoTokenizer
import evaluate
import seqeval
import spacy

# -> CONSTANTS
label2id = {
    'O': 0,
    -100: -100,
    'B-SINTOMA': 1,
    'I-SINTOMA': 2,
}

id2label = {
    0: 'O',
    1: 'B-SINTOMA',
    2: 'I-SINTOMA',
}


# -> PREPROCESS
def tokenize_and_align_bigbiokb(row: dict, label_mapper: dict, tokenizer) -> dict:
    """Gets a row of a bigbio_kb dataset, a tokenizer and a label lookup and
    outputs its HuggingFace form: IOB with labels aligned to subtokens.
    Splitted entities (and passages, if there were) are handled as separate entites. 
    For overlapping and nested entities the outermost span is preserved.
    """
    pos_in_text = 0
    tokenized_input = {
        'tokens': [],
        'labels': [],
        'input_ids': [],
        'attention_mask': [],
        'offsets': [],
        'text_labels': [],
    }
    
    # handle passages
    for passage in row['passages']:
        # handle passages split accross the text: treat each part as a separate sentence
        for h, text in enumerate(passage['text']):
            # catch offsets
            passage_start, passage_end = passage['offsets'][h][0], passage['offsets'][h][1]

            # tokenize passage text and remap the offsets for subtokens
            tokenized_passage = tokenizer(text,
                                          add_special_tokens=True,
                                          return_offsets_mapping=True,
                                          #is_split_into_words=True, 
                                          max_length=512,
                                          #padding='max_length',
                                          truncation=True
                                         )
            tokens = tokenized_passage.tokens()
            mapped_offsets = tokenized_passage.offset_mapping

            # store word_ids to label only the first token of each word
            word_ids = tokenized_passage.word_ids()

            # add to the offsets the length of all prior texts so the labels match
            for i, tup in enumerate(mapped_offsets):
                # Skip the first and last tuples (used for [CLS] and [SEP])
                if i == 0 or i == len(mapped_offsets) - 1:
                    continue
                mapped_offsets[i] = tuple(val + pos_in_text for val in tup)

            # handle entities - assign tagged labels to the tokens
            labels, text_labels = [0] * len(tokens), ['O'] * len(tokens)

            for entity in row['entities']:
                # handle entities split accross the text: treat each part as a separate entity
                for j, span in enumerate(entity['offsets']):
                    entity_start, entity_end = span[0], span[1]
                    # check the entity only if it falls within the passage offsets and is not mapped to 0 (ignored)
                    if entity_start in range(passage_start, passage_end) and label_mapper[f"B-{entity['type']}"] != 0:
                        for k, subtoken in enumerate(mapped_offsets):
                            if subtoken[0] == entity_start and subtoken != (0,0):
                                text_labels[k] = f'B-{entity["type"]}'
                                labels[k] = label_mapper[text_labels[k]]
                            elif subtoken[0] in range(entity_start, entity_end) and subtoken != (0,0): 
                                text_labels[k] = f'I-{entity["type"]}'
                                labels[k] = label_mapper[text_labels[k]]

            # make sure all subtokens are labeled as -100
            for k, subtoken in enumerate(mapped_offsets):
                if subtoken == (0,0) or word_ids[k] == word_ids[k-1]:
                    labels[k] = -100

            # update the marker of position in text with the length of the passage
            pos_in_text += (passage_end - passage_start) + 1

            # append results
            tokenized_input['tokens'].append(tokens)
            tokenized_input['labels'].append(labels)
            tokenized_input['text_labels'].append(text_labels)
            tokenized_input['input_ids'].append(tokenized_passage['input_ids'])
            tokenized_input['attention_mask'].append(tokenized_passage['attention_mask'])
            tokenized_input['offsets'].append(mapped_offsets)

    return tokenized_input


def tokenize_split(hf_split: datasets.arrow_dataset.Dataset, label_mapper: dict, model: str) -> dict:
    tokenized_split = {
        'tokens': [],
        'labels': [],
        'input_ids': [],
        'attention_mask': [],
        'offsets': [],
        'text_labels': [],
    }
    tokenizer = AutoTokenizer.from_pretrained(model)

    for row in tqdm.tqdm(hf_split, desc=f'Tokenizing dataset split'):
        tokenized_row = tokenize_and_align_bigbiokb(row, label_mapper, tokenizer)
        tokenized_split = {key: tokenized_split.get(key, []) + tokenized_row.get(key, []) for key in tokenized_split}
    return datasets.Dataset.from_dict(tokenized_split)


# -> SPAN MARKER
def bigbio2spanmarker(split: Dataset) -> Dataset:

    nlp = spacy.load("es_core_news_sm")

    output = {
        "filename": [],
        "document_id": [],
        "sentence_id": [],
        "tokens": [],
        "ner_tags": [],
        "text": [],        
    }
    
    for doc_id, row in tqdm.tqdm(enumerate(split), desc="Document progress:"):
        
        text = row["passages"][0]["text"][0]
        entities = row["entities"]
        doc = nlp(text)
        
        for sentence_id, sentence in enumerate(doc.sents):
            
            tokens = []
            token_positions = []
            
            for token in sentence:
                tokens.append(token.text)
                token_positions.append(token.idx)
        
            ner_tags = [0] * len(tokens)
                
            for i, position in enumerate(token_positions):
                for entity in entities:
                    for offset in entity["offsets"]:
                        if position==offset[0] or position in range(offset[0],offset[1]):
                            ner_tags[i] = 1

            output["filename"].append(row["document_id"])
            output["document_id"].append(doc_id)
            output["sentence_id"].append(sentence_id)
            output["tokens"].append(tokens)
            output["ner_tags"].append(ner_tags)
            output["text"].append(sentence.text)
        
    return datasets.Dataset.from_dict(output)
    
# -> EVALUATE
seqeval = evaluate.load("seqeval")
faireval = evaluate.load("hpi-dhc/FairEval")

def compute_fair(label_list: list) -> None:
    def _compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        fair_results = faireval.compute(predictions=true_predictions, references=true_labels)
        seqeval_results = seqeval.compute(predictions=true_predictions, references=true_labels, mode = "default")

        return {
            #'faireval': fair_results, 
            #'seqeval': seqeval_results, 
            'eval_overall_f1': seqeval_results['overall_f1'], 
            'eval_overall_fair_f1': fair_results['overall_f1']
        }

    return _compute_metrics

    
# -> TRAIN
def train(train_set: Dataset, eval_set: Dataset, model: str, args: dict) -> None:
    training_args = TrainingArguments(**args)
    tokenizer = AutoTokenizer.from_pretrained(model)
    trainer = Trainer(
        args = training_args,
        model = AutoModelForTokenClassification.from_pretrained(model, id2label = id2label, num_labels = len(id2label)),
        train_dataset= train_set,
        eval_dataset = eval_set,        
        tokenizer = tokenizer,
        data_collator = DataCollatorForTokenClassification(tokenizer),
        compute_metrics = compute_fair(list(id2label.values())),
    )
    trainer.train() 
    pass