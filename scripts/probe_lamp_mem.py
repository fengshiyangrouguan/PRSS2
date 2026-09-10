#!/usr/bin/env python
"""Locate the OOM stage in LaMP training (forward vs backward, which
component).  Prints torch.cuda.memory_allocated after each stage."""
import os
import sys
import torch

CCM_ROOT = "/root/autodl-tmp/third_party/ccm"
sys.path.insert(0, CCM_ROOT)
os.chdir(CCM_ROOT)


def show(tag):
    torch.cuda.synchronize()
    print(f"{tag}: allocated={torch.cuda.memory_allocated()/2**30:.2f}G "
          f"reserved={torch.cuda.memory_reserved()/2**30:.2f}G", flush=True)


def main():
    import train_lamp as T
    from types import SimpleNamespace

    args = SimpleNamespace(
        model_name_or_path="/root/autodl-tmp/llama-7b-hf",
        foundation="result/lamp/llama-7b-no",
        adapter="result/lamp/finetune/llama-7b-no-online-merge-ntok4",
        k=16, seed=0,
    )
    device = torch.device("cuda")
    show("start")
    model = T.build_official_host(args, device)
    show("after host")
    model.train()
    tokenizer = T.build_tokenizer(args.model_name_or_path)
    ds, collator = T.build_dataset_and_collator(tokenizer, model, args)
    show("after dataset")
    dataloader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=False, collate_fn=collator)
    batch = next(iter(dataloader))
    show("after collate (cpu)")
    batch = {k: v.to(device) for k, v in batch.items()}
    show("after to(device)")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape} {v.dtype}")
    with torch.cuda.amp.autocast(enabled=True):
        out = model(**batch)
    show("after fwd")
    print("loss =", float(out.loss))
    out.loss.backward()
    show("after bwd")
    grad_norm = sum((p.grad ** 2).sum() for p in model.parameters()
                    if p.grad is not None).sqrt()
    print("grad norm =", float(grad_norm))
    show("final")


if __name__ == "__main__":
    main()
