# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
from ipdb import set_trace
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "latents"])
MAX_N_LATENT = 8


import torch
from torch.nn.utils.rnn import pad_sequence


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []
        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)
        latents = inputs_embeds.new_zeros(
            (inputs_embeds.size(0), max_n_latents, inputs_embeds.size(-1))
        )
        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # replace it with the preceding last hidden states
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # 1) grab hidden state for the token that *preceded* the latent
                h = hidden_states[batch_idx,
                                token_idx - 1 - hidden_states_offset,
                                :]                       #   (H,)

                # 2) save it for the caller  ----------------
                latents[batch_idx, pass_idx, :] = h        # <<<<<<<<

                # 3) overwrite embedding so the LM can use it next time
                tensor_list[batch_idx][token_idx] = h

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits, latents=latents)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()
    def generate(
        self,
        input_ids,
        attention_mask=None,          # still unused
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs,
    ):
        device      = input_ids.device
        B, orig_len = input_ids.shape
        eos_id      = self.eos_token_id

        # ───── first forward pass (unchanged) ────────────────────────────────
        labels  = input_ids.clone()
        pos_ids = torch.arange(orig_len, device=device).expand(B, -1)
        outs    = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=device),
            labels,
            pos_ids,
        )
        outputs_latents = outs.latents
        inputs_embeds = outs.inputs_embeds                          # (B, L, D)

        # ───── buffers that hold ONLY the generated part ────────────────────
        gen_tokens  = [[] for _ in range(B)]     # python lists per sequence
        gen_latents = [[] for _ in range(B)]     # ditto for embeddings
        alive       = torch.ones(B, dtype=torch.bool, device=device)

        self.gen_forward_cnt = 0

        # ───── autoregressive loop ──────────────────────────────────────────
        for _ in range(max_new_tokens):
            logits = self.base_causallm(inputs_embeds=inputs_embeds).logits
            self.gen_forward_cnt += 1

            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)         # (B,)
            newly_finished = next_tokens.eq(eos_id)
            alive &= ~newly_finished

            # save tokens & latents for still-alive sequences
            if alive.any():
                next_tok_embeds = self.embedding(next_tokens)            # (B, D)
                for b in range(B):
                    if alive[b]:
                        gen_tokens[b].append(next_tokens[b].item())
                        gen_latents[b].append(next_tok_embeds[b])        # 1-D tensor

                # pad finished rows with zeros to keep tensor rectangular
                next_tok_embeds[~alive] = 0.0
                inputs_embeds = torch.cat(
                    (inputs_embeds, next_tok_embeds.unsqueeze(1)), dim=1
                )
            else:
                break

        # ───── FSDP sync (unchanged) ────────────────────────────────────────
        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=inputs_embeds)

        # ───── record prefix latents BEFORE the loop grows the tensor ------------
        prefix_latents = inputs_embeds.clone()          # (B, orig_len, D)

        # ... autoregressive loop stays exactly as in the previous version ...

        # ───── pack generated results --------------------------------------------
        max_gen_len = max(len(t) for t in gen_tokens) if max_new_tokens else 0
        if max_gen_len == 0:                                    # nothing generated
            tok_tensor = torch.empty(B, 0, dtype=torch.long,  device=device)
            lat_tensor = torch.empty(B, 0, prefix_latents.size(-1), device=device)
        else:
            D = prefix_latents.size(-1)

            # (B, Lgen) token IDs, eos-padded
            tok_tensor = torch.full(
                (B, max_gen_len), fill_value=eos_id, dtype=torch.long, device=device
            )
            # (B, Lgen, D) latent embeddings, zero-padded
            lat_tensor = torch.zeros(B, max_gen_len, D, device=device)

            for b in range(B):
                Lb = len(gen_tokens[b])
                if Lb:
                    tok_tensor[b, :Lb] = torch.tensor(gen_tokens[b], device=device)
                    lat_tensor[b, :Lb, :] = torch.stack(gen_latents[b], dim=0)

        # ───── return -------------------------------------------------------------
        if output_embedding:
            # 0: latents of the *autoregressive prefix* (input part)
            # 1: latents of the generated chunk
            # 2: generated token IDs
            return outputs_latents, tok_tensor
        else:
            return tok_tensor

    # def generate(
    #     self,
    #     input_ids,
    #     attention_mask,  # attention_mask is not used
    #     tokenizer,
    #     max_new_tokens=16,
    #     output_embedding=False,
    #     synced_gpus=False,
    #     **kwargs
    # ):

    #     self.gen_forward_cnt = 0

    #     assert input_ids.shape[0] == 1, "only support batch_size == 1 now"
    #     tokens = input_ids[0].detach().tolist()

    #     labels = input_ids.clone()  # placeholder. not used.
    #     outputs = self.forward(
    #         input_ids,
    #         torch.ones_like(input_ids, device=input_ids.device),
    #         labels,
    #         torch.arange(
    #             0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
    #         ).reshape(1, -1),
    #     )
    #     set_trace()
    #     inputs_embeds = outputs.inputs_embeds
    #     # get the first token using the current hidden state
    #     next_token = torch.argmax(outputs.logits[0, -1]).item()
    #     tokens.append(next_token)
    #     new_token_embed = self.embedding(
    #         torch.tensor(next_token, device=input_ids.device)
    #     ).view(1, 1, -1)
    #     new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

    #     # get other tokens
    #     for _ in range(max_new_tokens - 1):
    #         outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
    #         self.gen_forward_cnt += 1
    #         next_token = torch.argmax(outputs.logits[0, -1]).item()
    #         if next_token == self.eos_token_id:
    #             break
    #         tokens.append(next_token)
    #         new_token_embed = self.embedding(
    #             torch.tensor(next_token, device=input_ids.device)
    #         ).view(1, 1, -1)
    #         new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)
    #     if synced_gpus:
    #         # in FSDP, the number of forward pass need to be the same across devices
    #         while (
    #             self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
    #         ):  # leave some room for latent tokens
    #             self.gen_forward_cnt += 1
    #             _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

    #     if output_embedding:
    #         # for analysis purpose
    #         return torch.tensor(tokens).view(1, -1), new_inputs_embeds

    #     else:
    #         return torch.tensor(tokens).view(1, -1)