"""
 Copyright (c) 2024, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import os
import json
import argparse
import pickle
from typing import Optional, Union, List, Tuple

import datasets
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    # Trainer,
    # TrainingArguments,
    # StoppingCriteriaList
    )

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

from trl import (
    SFTConfig, 
    SFTTrainer, 
    DataCollatorForCompletionOnlyLM
)

from peft import (
    LoraConfig, 
    # get_peft_model, 
    # prepare_model_for_kbit_training, 
    PeftModel
    )
# from transformers import BitsAndBytesConfig

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from utils import extract_python_code


class LlamaForCausalLMWithReward(AutoModelForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        rewards: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none' if rewards is not None else 'mean')
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if rewards is not None:
                rewards = rewards.to(loss.device)
                loss *= rewards
                loss = loss.mean()
                

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class GemmaForCausalLMWithReward(AutoModelForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        rewards: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, GemmaForCausalLM

        >>> model = GemmaForCausalLM.from_pretrained("google/gemma-7b")
        >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")

        >>> prompt = "What is your favorite condiment?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "What is your favorite condiment?"
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # if labels is None and not is_torchdynamo_compiling():
        #     logger.warning_once(
        #         "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
        #     )
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # TODO: remove the float() operation in v4.46
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none' if rewards is not None else 'mean')
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if rewards is not None:
                rewards = rewards.to(loss.device)
                loss *= rewards
                loss = loss.mean()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        

# from pdb import set_trace; set_trace()
parser = argparse.ArgumentParser(description='Train a model for Visual Reasoning')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--train_kwargs', type=str, default="{}")
parser.add_argument('--model_kwargs', type=str, default="{}")
parser.add_argument('--data_path', type=str, default='', help='Precomputed Data path')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--iter', type=int, default=0)
parser.add_argument('--output_dir', type=str, default='.')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--num_improve_steps', type=int, default=1)
parser.add_argument('--train_prompt_file', type=str, default='.')
args = parser.parse_args()

output_dir = args.output_dir
data_path = args.data_path

os.makedirs(output_dir, exist_ok=True)
adapter_ckpt_dir = os.path.join(output_dir, 'adapter_ckpts')
os.makedirs(adapter_ckpt_dir, exist_ok=True)

# [{question, pred_code, unit_test_results:[{acc}]}]
input_dataset = pickle.load(open(os.path.join(args.data_path), 'rb'))

train_kwargs = json.loads(args.train_kwargs)
model_kwargs = json.loads(args.model_kwargs)

train_prompt = open(args.train_prompt_file).read()
def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['label'])):
            text = f"{train_prompt.replace('INSERT_QUERY_HERE',example['question'][i])}{extract_python_code(example['label'][i])}"
            output_texts.append(text)
        return output_texts
iter = args.iter
unit_test_reward = train_kwargs.get('unit_test_reward', 'ratio')
unit_test_reward_weight = train_kwargs.get('unit_test_reward_weight', 1.0)
only_correct = train_kwargs.get('only_correct', False)
if unit_test_reward!='none':
    for d in input_dataset:
        d['reward'] = 0.
        if only_correct and d['true_accuracy'] < 1.:
            continue
        acc = float(d['unit_test_results']['acc'])
        if unit_test_reward=='all' or 'all' in unit_test_reward:
            d['reward'] = 1.0 if acc == 1.0 else d['reward']
        if unit_test_reward=='ratio' or 'ratio' in unit_test_reward:
            d['reward'] = acc
        if unit_test_reward=='threshold' or 'threshold' in unit_test_reward:
            d['reward'] = 1. if  acc>=train_kwargs.get('threshold', 0.5) else d['reward']
        if unit_test_reward == 'threshold_correct' or 'threshold_correct' in unit_test_reward:
            d['reward'] = 1. if  acc>=train_kwargs.get('threshold', 0.5) and acc==1. else d['reward']
else:
    for d in input_dataset:
        if only_correct and d['true_accuracy'] < 1.:
            d['reward'] = 0.
            continue
        d['reward'] = 1.0
# from pdb import set_trace; set_trace()


questions, pred_codes, rewards = zip(*[(d['question'], d['pred_code'], d['reward']*unit_test_reward_weight) for d in input_dataset])
improve_dataset = datasets.Dataset.from_dict({'question':questions, 'label':pred_codes, 'reward': rewards})
improve_dataset = improve_dataset.filter(lambda x: x['reward']>0.)
print('Training dataset size:', len(improve_dataset))

# setup tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, 
                                          padding='longest', 
                                          use_fast=True,
                                          truncation=True,
                                          trust_remote_code=True,
                                          token=os.getenv('HF_ACCESS_TOKEN'))
tokenizer.pad_token = tokenizer.eos_token

response_template = "\nProgram:"
collator = DataCollatorForCompletionOnlyLM(response_template=tokenizer.encode(response_template, add_special_tokens = False)[-2:], tokenizer=tokenizer)



# from virep parameters
lora_kwargs = train_kwargs.get('lora_kwargs', {})
lora_config = LoraConfig(
            r=lora_kwargs.get('r', 16),
            target_modules = lora_kwargs.get('target_modules', 
                                [
                                    "k_proj",
                                    "v_proj",
                                    "q_proj",
                                    "o_proj"
                                    ]),
            lora_alpha=lora_kwargs.get('lora_alpha', 32),
            lora_dropout=lora_kwargs.get('lora_dropout', 0.05),
            bias=lora_kwargs.get('bias', 'none'),
            task_type="CAUSAL_LM"
            )


if iter>0:
    if 'llama' in args.model_name.lower():
        model = LlamaForCausalLMWithReward.from_pretrained(args.model_name,
                                                    torch_dtype=torch.bfloat16, # training in bfloat is more stable https://huggingface.co/docs/transformers/en/model_doc/llama2#usage-tips
                                                    # device_map="auto",
                                                    trust_remote_code=True,
                                                    token=os.getenv('HF_ACCESS_TOKEN'),
                                                    )
    elif 'gemma' in args.model_name.lower():
        model = GemmaForCausalLMWithReward.from_pretrained(args.model_name,
                                                    torch_dtype=torch.bfloat16, # training in bfloat is more stable https://huggingface.co/docs/transformers/en/model_doc/llama2#usage-tips
                                                    # device_map="auto",
                                                    trust_remote_code=True,
                                                    token=os.getenv('HF_ACCESS_TOKEN'),
                                                    )
    model.enable_input_require_grads()
    model = PeftModel.from_pretrained(model, os.path.join(adapter_ckpt_dir, f"model_{iter-1}"), is_trainable=True)        
else:
    max_steps = 0
    if 'llama' in args.model_name.lower():
        model = LlamaForCausalLMWithReward.from_pretrained(args.model_name,
                                                    torch_dtype=torch.bfloat16, # training in bfloat is more stable https://huggingface.co/docs/transformers/en/model_doc/llama2#usage-tips
                                                    # device_map="auto",
                                                    trust_remote_code=True,
                                                    token=os.getenv('HF_ACCESS_TOKEN'),
                                                    )
    elif 'gemma' in args.model_name.lower():
        model = GemmaForCausalLMWithReward.from_pretrained(args.model_name,
                                                    torch_dtype=torch.bfloat16, # training in bfloat is more stable https://huggingface.co/docs/transformers/en/model_doc/llama2#usage-tips
                                                    # device_map="auto",
                                                    trust_remote_code=True,
                                                    token=os.getenv('HF_ACCESS_TOKEN'),
                                                    )
    
num_devices = torch.cuda.device_count()

if iter>0:
    last_model = os.path.join(adapter_ckpt_dir, f"model_{iter-1}")
    model.load_adapter(last_model, adapter_name='model_adapter')

training_args = SFTConfig(
        output_dir=os.path.join(output_dir, f'train'),          # output directory
        per_device_train_batch_size=args.batch_size,  # batch size for training
        logging_dir=os.path.join(output_dir, f'logs'),            # directory for storing logs
        logging_steps=1,
        bf16=True,
        # max_steps=max_steps,
        # save_steps=max_steps,
        seed=args.seed,
        save_strategy='epoch',
        report_to='tensorboard',
        warmup_ratio=train_kwargs.get('warmup_ratio', 0.1),
        max_grad_norm = train_kwargs.get('max_grad_norm', 1.0),
        lr_scheduler_type=train_kwargs.get('lr_scheduler_type', 'constant'),
        learning_rate=train_kwargs.get('learning_rate', 1e-4),
        num_train_epochs=train_kwargs.get('num_improve_steps', 1)*(iter+1),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        max_seq_length=4096,
        remove_unused_columns=True
    )

tokenizer.padding_side='right' # for training
trainer = SFTTrainer(
    model=model,
    train_dataset=improve_dataset,
    tokenizer=tokenizer,
    formatting_func=formatting_prompts_func,
    args=training_args,
    data_collator=collator,
    packing=False,
    peft_config=lora_config if iter==0 else None,
    )
print(f"Trainable parameters: {trainer.get_num_trainable_parameters()}")
trainer.train(resume_from_checkpoint=iter!=0)
trainer.model.save_pretrained(os.path.join(adapter_ckpt_dir, f'model_{iter}'))

