from datasets import load_dataset
import argparse
import sys
import shutil
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
import logging
from tqdm import tqdm, trange
from transformers import (AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator,
                          get_scheduler)

from utils import set_seed, freeze_model, prepare_data
from constants import SQUAD_VERSION

logging.basicConfig(level=logging.INFO,
                    format='[distillation:%(levelname)s] %(message)s')


def prepare_models(args):
    torch_dtype = torch.float16 if args['fp16'] else torch.float32
    model = AutoModelForQuestionAnswering.from_pretrained(args['student_model_path'], torch_dtype=torch_dtype)
    teacher_model = AutoModelForQuestionAnswering.from_pretrained(args['teacher_model_path'], torch_dtype=torch_dtype)
    freeze_model(teacher_model)
    return model, teacher_model


def prepare_checkpoint_directory(args):
    if os.path.exists(args['checkpoint_dir']):
        if args['overwrite_checkpoint']:
            logging.warning(f'Overwriting checkpoint directory: {args["checkpoint_dir"]}')
            shutil.rmtree(args["checkpoint_dir"])
        else:
            raise ValueError(f'Checkpoint directory already exists ({args["checkpoint_dir"]}) but '
                             f'overwriting is disabled. Use overwrite_checkpoint = True for overwriting.')
    else:
        os.makedirs(args['checkpoint_dir'])


def train(args, dataset, model, tokenizer, teacher=None):
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset,
                                  batch_size=args['batch_size'],
                                  sampler=train_sampler,
                                  collate_fn=DefaultDataCollator())

    if args['max_steps'] > 0:
        num_training_steps = args['max_steps']
        args['num_epochs'] = args['max_steps'] // (len(train_dataloader) // args['gradient_accumulation_steps']) + 1
    else:
        num_training_steps = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_epochs']

    # Report training info
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(dataset))
    logging.info("  Num Epochs = %d", args['num_epochs'])
    logging.info("  Total optimization steps = %d", num_training_steps)

    optimizer = AdamW(model.parameters(), lr=args['learning_rate'])
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    if args['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model.float(), optimizer, opt_level='O1')

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    train_iterator = trange(int(args['num_epochs']), desc="Epoch")
    set_seed(args['seed'])

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            if teacher is not None:
                teacher.eval()

            inputs = {'input_ids': batch['input_ids'].to(model.device),
                      'attention_mask': batch['attention_mask'].to(model.device),
                      'start_positions': batch['start_positions'].to(model.device),
                      'end_positions': batch['end_positions'].to(model.device)}

            outputs = model(**inputs, return_dict=True, output_hidden_states=True)
            loss, start_logits, end_logits, hidden_states = outputs.loss, outputs.start_logits, outputs.end_logits, outputs.hidden_states[-1]

            # Get distillation loss using the teacher model
            if teacher is not None:
                with torch.no_grad():
                    outputs_teacher = teacher(input_ids=batch['input_ids'].to(teacher.device),
                                              attention_mask=batch['attention_mask'].to(teacher.device),
                                              return_dict=True,
                                              output_hidden_states=True)
                    start_logits_teacher, end_logits_teacher, hidden_states_teacher = outputs_teacher.start_logits, outputs_teacher.end_logits, outputs_teacher.hidden_states[-1]

                assert start_logits_teacher.size() == start_logits.size()
                assert end_logits_teacher.size() == end_logits.size()
                assert hidden_states_teacher.size() == hidden_states.size()

                # Calculate distillation loss (start and end logits)
                loss_distill_func = nn.KLDivLoss(reduction='batchmean')
                loss_start = loss_distill_func(F.log_softmax(start_logits / args['temperature'], dim=-1),
                                               F.softmax(start_logits_teacher / args['temperature'], dim=-1)) * (args['temperature'] ** 2)
                loss_end = loss_distill_func(F.log_softmax(end_logits / args['temperature'], dim=-1),
                                             F.softmax(end_logits_teacher / args['temperature'], dim=-1)) * (args['temperature'] ** 2)
                loss_distill = (loss_start + loss_end) / 2.

                # Calculate hidden states loss (last layer hidden representations) as cosine distance.
                # Target is a vector of ones
                loss_hidden_states_func = nn.CosineEmbeddingLoss(reduction='sum')
                loss_hidden_states = loss_hidden_states_func(hidden_states[:, 0, :] / args['temperature'],
                                                             hidden_states_teacher[:, 0, :] / args['temperature'],
                                                             target=torch.ones(hidden_states.size(0)).to(hidden_states.device))

                loss = args['alpha'] * loss_distill + args['beta'] * loss + args['gamma'] * loss_hidden_states

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

            logging.info(f'Epoch {epoch+1}, Step {step+1} - Loss: {loss.item()}')

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

            if 0 < args['max_steps'] < global_step:
                epoch_iterator.close()
                break

        # Save model checkpoint
        output_dir = os.path.join(args['checkpoint_dir'], f'checkpoint-epoch-{epoch}')
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Saving model checkpoint to %s", output_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))

        if 0 < args['max_steps'] < global_step:
            train_iterator.close()
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/distillation1.json",
                        required=True,
                        help='Path to the json config with the parameters for the distillation process')

    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.config_path):
        raise ValueError(f'Path to config {args.config_path} does not exist.')

    logging.info(f"Loading parameters for distillation process from: {args.config_path}")
    with open(args.config_path) as file:
        try:
            config_args = json.load(file)
        except Exception as e:
            logging.error(e)

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(config_args['teacher_model_path'])

    if not tokenizer.is_fast:
        raise ValueError('Only fast tokenizers are supported.')

    squad_dataset = load_dataset("squad_es", SQUAD_VERSION)
    train_dataset = prepare_data(squad_dataset["train"], tokenizer, config_args)

    # Prepare model and teacher model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, teacher_model = prepare_models(config_args)
    model.to(device)
    teacher_model.to(device)

    # Prepare checkpoint output directory
    prepare_checkpoint_directory(config_args)

    # Run distillation
    train(config_args, train_dataset, model, tokenizer, teacher=teacher_model)
