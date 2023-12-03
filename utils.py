import torch
import random
from functools import partial
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def freeze_model(model_to_freeze):
    for param in model_to_freeze.parameters():
        param.requires_grad = False


def preprocess_instances(instances, tokenizer, max_length, stride):
    questions = [q.strip() for q in instances["question"]]
    inputs = tokenizer(
        questions,
        instances["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    answers = instances["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(inputs["offset_mapping"]):
        sample_idx = sample_map[i]
        example_ids.append(instances["id"][sample_idx])
        answer = answers[sample_idx]
        if len(answer["text"]) == 0:
            # Impossible question
            start_positions.append(0)
            end_positions.append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["example_id"] = example_ids
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def prepare_data(dataset, tokenizer, args):
    train_dataset = dataset.map(
        partial(preprocess_instances,
                tokenizer=tokenizer,
                max_length=args['max_length'],
                stride=args['stride']),
        batched=True,
        remove_columns=dataset.column_names,
    )

    return train_dataset


def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()
