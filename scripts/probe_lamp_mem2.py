#!/usr/bin/env python
"""Per-layer memory probe: hook every decoder layer, print allocated
after each layer's forward, so we can see WHERE the memory explodes."""
import os
import sys
import torch

CCM_ROOT = "/root/autodl-tmp/third_party/ccm"
sys.path.insert(0, CCM_ROOT)
os.chdir(CCM_ROOT)


def main():
    import train_lamp as T
    from types import SimpleNamespace

    args = SimpleNamespace(
        model_name_or_path="/root/autodl-tmp/llama-7b-hf",
        foundation="result/lamp/llama-7b-no",
        adapter="result/lamp/finetune/llama-7b-no-online-merge-ntok4",
        k=16, seed=0, fp16_weights=True,
    )
    device = torch.device("cuda")
    model = T.build_official_host(args, device)
    model.train()
    tokenizer = T.build_tokenizer(args.model_name_or_path)
    ds, collator = T.build_dataset_and_collator(tokenizer, model, args)
    dataloader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=False, collate_fn=collator)
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    print("batch keys:", list(batch.keys()))
    print("T =", batch["input_ids"].shape)

    layers = model.base_model.model.model.layers

    def make_hook(i):
        def hook(m, args, out):
            torch.cuda.synchronize()
            print(f"  layer {i:2d} after fwd: "
                  f"alloc={torch.cuda.memory_allocated()/2**30:.2f}G",
                  flush=True)
        return hook

    handles = [l.register_forward_hook(make_hook(i))
               for i, l in enumerate(layers)]
    try:
        with torch.cuda.amp.autocast(enabled=True):
            out = model(**batch)
        torch.cuda.synchronize()
        print(f"final alloc={torch.cuda.memory_allocated()/2**30:.2f}G "
              f"loss={float(out.loss):.4f}", flush=True)
        out.loss.backward()
        torch.cuda.synchronize()
        print(f"after bwd alloc={torch.cuda.memory_allocated()/2**30:.2f}G",
              flush=True)
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
