import collections
import argparse
import sys
import os
import json
import logging
from functools import partial
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
import evaluate

from constants import SQUAD_VERSION
from utils import prepare_data

logging.basicConfig(level=logging.INFO,
                    format='[inference:%(levelname)s] %(message)s')


def inference_collate_func(data, device='cpu'):
    input_ids = torch.stack([torch.as_tensor(sample['input_ids']) for sample in data]).to(device)
    attention_mask = torch.stack([torch.as_tensor(sample['attention_mask']) for sample in data]).to(device)
    start_positions = torch.stack([torch.as_tensor(sample['start_positions']) for sample in data]).to(device)
    end_positions = torch.stack([torch.as_tensor(sample['end_positions']) for sample in data]).to(device)

    assert len(input_ids) == len(attention_mask) == len(start_positions) == len(end_positions)

    return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': start_positions,
            'end_positions': end_positions}

@torch.no_grad()
def run_inference(args, model, data_loader, dataset):
    validation_start_logits = np.empty((len(data_loader.dataset), args['max_length']))
    validation_end_logits = np.empty((len(data_loader.dataset), args['max_length']))

    for batch_id, batch in tqdm(enumerate(data_loader), desc="Obtaining start/end logits"):
        with torch.no_grad():
            outputs = model(**batch, return_dict=True, output_hidden_states=True)
            start_logits = outputs.start_logits.cpu().numpy()
            end_logits = outputs.end_logits.cpu().numpy()
            assert len(start_logits) == len(end_logits)
            offset = batch_id * len(start_logits)
            validation_start_logits[offset:offset + len(start_logits)] = start_logits
            validation_end_logits[offset:offset + len(end_logits)] = end_logits

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(validation_dataset):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(dataset, desc=f"Obtaining and formatting answers in "
                                      f"validation set (n_best={args['n_best']})"):
        example_id = example["id"]
        context = example["context"]
        answers = []

        for feature_index in example_to_features[example_id]:
            start_logit = validation_start_logits[feature_index]
            end_logit = validation_end_logits[feature_index]
            offsets = validation_dataset["offset_mapping"][feature_index]

            start_indexes = np.argsort(start_logit)[-1: -args['n_best'] - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -args['n_best'] - 1: -1].tolist()
            for start_index in start_indexes:
                if offsets[start_index] is None:
                    continue

                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length.
                    if (
                            end_index < start_index
                            or end_index - start_index + 1 > args['max_answer_length']
                    ):
                        continue

                    answers.append(
                        {
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                    )

        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
        else:
            best_answer = {"text": ""}

        predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})

    return predicted_answers


def calculate_metrics(metric, squad_dataset, predicted_answers):
    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in squad_dataset
    ]

    metrics = metric.compute(predictions=predicted_answers, references=theoretical_answers)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/inference1.json",
                        required=True,
                        help='Path to the json config with the parameters for running evaluation of a model on Squad')

    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.config_path):
        raise ValueError(f'Path to config {args.config_path} does not exist.')

    logging.info(f"Loading parameters for inference from: {args.config_path}")
    with open(args.config_path) as file:
        try:
            config_args = json.load(file)
        except Exception as e:
            logging.error(e)

    # Prepare dataset
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = 'mps'
    tokenizer = AutoTokenizer.from_pretrained(config_args['model_path'])

    if not tokenizer.is_fast:
        raise ValueError('Only fast tokenizers are supported.')

    squad_dataset = load_dataset("squad_es", SQUAD_VERSION)["validation"]
    validation_dataset = prepare_data(squad_dataset, tokenizer, config_args)
    val_dataloader = DataLoader(validation_dataset, batch_size=config_args['batch_size'], shuffle=False,
                                collate_fn=partial(inference_collate_func, device=device))

    # Prepare model
    model = AutoModelForQuestionAnswering.from_pretrained(config_args['model_path'],
                                                          torch_dtype=torch.float32).to(device)
    model.eval()

    # Run predictions
    predicted_answers = run_inference(config_args, model, val_dataloader, squad_dataset)

    # Calculate metrics
    metric = evaluate.load("squad")
    metrics = calculate_metrics(metric, squad_dataset, predicted_answers)
    logging.info(f'Metrics: {metrics}')
