import torch
from coconut import Coconut
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
import argparse
import os
from dataset import get_dataset, MyCollator
import vllm
from torch.utils.data.distributed import DistributedSampler
from ipdb import set_trace
from math_verify import parse,verify
from torch import optim

def get_question_latent_dataset(
    scheduled_stage,
    base_dataset_valid,
    start_id,
    latent_id,
    end_id,
    no_special_marker=False,
):

    def process_dataset(sample):

        max_latent_stage = 3
        k = min(max_latent_stage, scheduled_stage)

        k *= 2

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }

    return base_dataset_valid.map(
        process_dataset, remove_columns=list(base_dataset_valid.features), num_proc=32
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="coconut")
    parser.add_argument("--model_id", type=str)

    ckpt_path = "/n/netscratch/kempner_sham_lab/Everyone/ameterez/coconut/coconut_ckpt/20250627_coconut_gsm_n24my/checkpoint_9"

    # model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    configs = parser.parse_args()
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    torch.distributed.barrier()

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens("<|start-latent|>")
    tokenizer.add_tokens("<|end-latent|>")
    tokenizer.add_tokens("<|latent|>")
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    saved_weights = torch.load(ckpt_path, map_location=torch.device(rank))

    model.resize_token_embeddings(len(tokenizer))
    embeddings = model.get_input_embeddings()
    target_id = tokenizer.convert_tokens_to_ids("<<")
    # initialize the new token embeddings with a known token
    # it helps stablize the training
    for token_id in [latent_id, start_id, end_id]:
        target_embedding = embeddings.weight.data[token_id]
        embeddings.weight.data[token_id] = target_embedding
        # The input embeddings and lm heads are tied in GPT2. So the code below is not necessary
        lm_head = model.lm_head
        lm_head.weight.data[token_id] = lm_head.weight.data[target_id]

    model = Coconut(model, latent_id, start_id, end_id, tokenizer.eos_token_id)
    print(model.load_state_dict(saved_weights, strict=False))
    print(f"Running FSDP on rank = {rank}, world size = {world_size}")
    model = model.to(rank)
    # model.to(torch.bfloat16)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=1e-3,
    )
    print(model)


    val_path = "/n/netscratch/kempner_sham_lab/Everyone/ameterez/coconut/data/gsm_valid.json"
    base_dataset_valid = get_dataset(
        val_path, tokenizer, 100000000
    )
    scheduled_stage = 2
    dataset_gen_val = get_question_latent_dataset(
        scheduled_stage,
        base_dataset_valid,
        start_id,
        latent_id,
        end_id,
        no_special_marker=False,
    )
    collator = MyCollator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)
    valid_gen_dataloader = torch.utils.data.DataLoader(
        dataset_gen_val,
        num_workers=1,
        pin_memory=True,
        batch_size=32,
        drop_last=True,
        collate_fn=collator,
        sampler=DistributedSampler(dataset_gen_val, shuffle=False),
    )

    running_baseline = None
    for idx, batch in enumerate(valid_gen_dataloader):
        test_idx = batch["idx"]
        answers = base_dataset_valid[batch["idx"]]['answer']
        batch = {
            k: v.to(rank)
            for k, v in batch.items()
            if v != None and k not in ["idx", "position_ids"]
        }

        latents, tokens = model.generate(
            **batch,
            max_new_tokens=16,
            tokenizer=tokenizer,
            synced_gpus=False,
            output_embedding=True
        )

        decoded_tokens = tokenizer.batch_decode(tokens)
        with torch.no_grad():
            reward = torch.tensor([verify(parse(decoded_tokens[i]), answers[i])*1.0 for i in range(len(decoded_tokens))]).to(rank).unsqueeze(1)
            if running_baseline is None:
                running_baseline = reward
            running_baseline = 0.9*running_baseline + 0.1*reward.mean()
        advantage = reward - running_baseline                        # (batch,)
        beta     = 0.9
        weights  = torch.exp(advantage / beta)          # (batch,)

        print(decoded_tokens)
        loss = (((latents[:, 1:, :] - latents[:, :-1, :]).norm(dim=-1)) * weights.detach()).sum(-1)
        loss = loss.mean()
        print(f"Loss: {loss.item()}, Average reward: {reward.mean()}")

        model.zero_grad()
        loss.backward()
        optimizer.step()