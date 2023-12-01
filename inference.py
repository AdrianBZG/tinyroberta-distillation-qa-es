from datasets import load_dataset
import torch
import numpy as np
import collections
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator
import evaluate
from torch.utils.data import DataLoader

from constants import SQUAD_VERSION

EVAL_BATCH_SIZE = 50

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
max_length = 384
stride = 128
n_best = 1
max_answer_length = 30
squad = load_dataset("squad_es", SQUAD_VERSION)
metric = evaluate.load("squad")
MODEL_NAME = "stevemobs/roberta-large-fine-tuned-squad-es"  # F1 75.9
MODEL_NAME = "checkpoints/tinyroberta-squad-es/checkpoint-epoch-0"
tokenizer = AutoTokenizer.from_pretrained("stevemobs/roberta-large-fine-tuned-squad-es")
if not tokenizer.is_fast:
    raise ValueError('Only fast tokenizers are supported.')
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
model.eval()

def preprocess_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(inputs["offset_mapping"]):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
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


def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


small_validation_subset = squad["validation"].select(range(200))
validation_dataset = small_validation_subset.map(
    preprocess_examples,  # preprocess_validation_examples
    batched=True,
    remove_columns=squad["validation"].column_names,
)

def eval_collate_fn(data):
    input_ids = torch.stack([torch.as_tensor(sample['input_ids']) for sample in data]).to(device)
    attention_mask = torch.stack([torch.as_tensor(sample['attention_mask']) for sample in data]).to(device)
    start_positions = torch.stack([torch.as_tensor(sample['start_positions']) for sample in data]).to(device)
    end_positions = torch.stack([torch.as_tensor(sample['end_positions']) for sample in data]).to(device)

    assert len(input_ids) == len(attention_mask) == len(start_positions) == len(end_positions)

    return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_positions': start_positions,
            'end_positions': end_positions}


val_dataloader = DataLoader(validation_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=eval_collate_fn)
validation_start_logits = np.empty((len(val_dataloader.dataset), max_length))
validation_end_logits = np.empty((len(val_dataloader.dataset), max_length))

for batch_id, batch in tqdm(enumerate(val_dataloader), desc="Obtaining validation start/end logits"):
    with torch.no_grad():
        outputs = model(**batch, return_dict=True, output_hidden_states=True)
        start_logits = outputs.start_logits.cpu().numpy()
        end_logits = outputs.end_logits.cpu().numpy()
        assert len(start_logits) == len(end_logits)
        offset = batch_id*len(start_logits)
        validation_start_logits[offset:offset + len(start_logits)] = start_logits
        validation_end_logits[offset:offset + len(end_logits)] = end_logits


example_to_features = collections.defaultdict(list)
for idx, feature in enumerate(validation_dataset):
    example_to_features[feature["example_id"]].append(idx)


predicted_answers = []
for example in tqdm(small_validation_subset, desc=f"Obtaining and formatting answers in "
                                                  f"validation set (n_best={n_best})"):
    example_id = example["id"]
    context = example["context"]
    answers = []

    for feature_index in example_to_features[example_id]:
        start_logit = validation_start_logits[feature_index]
        end_logit = validation_end_logits[feature_index]
        offsets = validation_dataset["offset_mapping"][feature_index]

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
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
                    or end_index - start_index + 1 > max_answer_length
                ):
                    continue

                answers.append(
                    {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                )

    if len(answers) > 0:
        best_answer = max(answers, key=lambda x: x["logit_score"])
    else:
        best_answer = {"text": ""}
    predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})


theoretical_answers = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in small_validation_subset
]

print(predicted_answers[0])
print(theoretical_answers[0])

print("Calculating metric")
print(metric.compute(predictions=predicted_answers, references=theoretical_answers))
