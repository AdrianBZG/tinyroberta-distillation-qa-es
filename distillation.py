from datasets import load_dataset
import argparse
import sys
import shutil
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
import logging
from tqdm import tqdm, trange
from transformers import (AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, DefaultDataCollator,
                          get_scheduler)

from modeling import CustomRobertaForQuestionAnswering
from utils import set_seed, freeze_model, prepare_data, soft_cross_entropy
from constants import SQUAD_VERSION

logging.basicConfig(level=logging.INFO,
                    format='[distillation:%(levelname)s] %(message)s')


def prepare_models(args):
    teacher_model = CustomRobertaForQuestionAnswering.from_pretrained(args['teacher_model_path'])
    model = CustomRobertaForQuestionAnswering.from_pretrained(args['student_model_path'],
                                                              fit_size=teacher_model.config.hidden_size)
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

    num_warmup_steps = int(num_training_steps * args['warmup_proportion'])

    # Report training info
    logging.info("***** Running training *****")
    logging.info("  Num examples = %d", len(dataset))
    logging.info("  Num Epochs = %d", args['num_epochs'])
    logging.info("  Warmup steps = %d", num_warmup_steps)
    logging.info("  Total optimization steps = %d", num_training_steps)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['eps'])
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    global_step = 0
    tr_loss = 0.0

    scaler = GradScaler()

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

            with torch.autocast(device_type="cuda", dtype=torch.float16 if args['fp16'] else torch.float32):
                outputs = model(**inputs, return_dict=True, output_hidden_states=True, output_attentions=True, is_student=True)
                loss, start_logits, end_logits, hidden_states, attentions, seq_output = outputs.loss, outputs.start_logits, outputs.end_logits, outputs.hidden_states[-1], outputs.attentions, outputs.sequence_output

                # Get distillation losses using the teacher model
                if teacher is not None:
                    attention_loss = 0.
                    hidden_loss = 0.

                    loss_states_func = nn.MSELoss()

                    with torch.no_grad():
                        outputs_teacher = teacher(input_ids=batch['input_ids'].to(teacher.device),
                                                  attention_mask=batch['attention_mask'].to(teacher.device),
                                                  return_dict=True,
                                                  output_hidden_states=True,
                                                  output_attentions=True)
                        start_logits_teacher, end_logits_teacher, hidden_states_teacher, attentions_teacher, seq_output_teacher = outputs_teacher.start_logits, outputs_teacher.end_logits, outputs_teacher.hidden_states[-1], outputs_teacher.attentions, outputs_teacher.sequence_output

                    teacher_layer_num = len(attentions_teacher)
                    student_layer_num = len(attentions)
                    assert teacher_layer_num % student_layer_num == 0

                    # Attention loss
                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    new_teacher_atts = [attentions_teacher[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]
                    assert len(attentions) == len(new_teacher_atts)
                    assert new_teacher_atts[0].size() == attentions[0].size()
                    for student_att, teacher_att in zip(attentions, new_teacher_atts):
                        student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(student_att.device),
                                                  student_att)
                        teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(teacher_att.device),
                                                  teacher_att)

                        tmp_loss = loss_states_func(student_att, teacher_att)
                        attention_loss += tmp_loss

                    # Representation loss
                    new_teacher_seq_output = [seq_output_teacher[i * layers_per_block] for i in range(student_layer_num + 1)]
                    assert len(new_teacher_seq_output) == len(seq_output)
                    assert new_teacher_seq_output[0].size() == seq_output[0].size()
                    for student_seqout, teacher_seqout in zip(seq_output, new_teacher_seq_output):
                        tmp_loss = loss_states_func(student_seqout, teacher_seqout)
                        hidden_loss += tmp_loss

                    assert start_logits_teacher.size() == start_logits.size()
                    assert end_logits_teacher.size() == end_logits.size()

                    # Calculate distillation loss (start and end logits)
                    loss_start = soft_cross_entropy(start_logits / args['temperature'],
                                                    start_logits_teacher / args['temperature'])

                    loss_end = soft_cross_entropy(end_logits / args['temperature'],
                                                  end_logits_teacher / args['temperature'])

                    loss_distill = loss_start + loss_end

                    loss = loss_distill + loss + attention_loss + hidden_loss

            if args['fp16']:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

            is_doing_warmup = global_step < num_warmup_steps
            logging.info(f'Epoch {epoch+1}/{args["num_epochs"]}, Step {step+1}, Global: {global_step}/{num_training_steps} - Warmup: {"Yes" if is_doing_warmup else "No"} - Loss: {loss.item()}')

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
        tokenizer.save_pretrained(output_dir)

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
