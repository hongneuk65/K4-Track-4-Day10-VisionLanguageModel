"""GPU operations. Imported only when a training/inference command is run."""
import gc
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path

from lab.data import RoadDataset, collate_examples, encode_example, subset_train, user_message
from scripts.labcommon import read_jsonl

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LLM_TARGET = r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
VIT_TARGET = r"model\.visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.(?:linear_fc1|linear_fc2))"


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def projector_name(name):
    return "visual.merger." in name or "visual.deepstack_merger_list." in name


def architecture_counts(model):
    """Count logical parameters rather than packed 4-bit storage elements."""
    counts = {"llm_including_embedding": 0, "vit_without_projector": 0, "projector": 0, "other": 0}
    for name, p in model.named_parameters():
        state = getattr(p, "quant_state", None)
        count = math.prod(state.shape) if state is not None else p.numel()
        key = ("projector" if projector_name(name) else "vit_without_projector" if "visual." in name
               else "llm_including_embedding" if "language_model." in name else "other")
        counts[key] += count
    counts["total"] = sum(counts.values())
    return counts


def environment():
    import importlib.metadata
    import platform
    import torch
    versions = {}
    for name in ("torch", "torchvision", "transformers", "peft", "bitsandbytes", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": platform.python_version(), "packages": versions,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def load_base():
    import torch
    from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("This run requires CUDA. Select T4 GPU in the Colab runtime settings.")
    config = AutoConfig.from_pretrained(MODEL_ID)
    # Use exact leaf names: some Transformers versions do not interpret 'visual'
    # as a subtree exclusion. Meta construction allocates no model weights.
    with torch.device('meta'):
        skeleton = Qwen3VLForConditionalGeneration(config)
    skip_modules = [name for name, module in skeleton.named_modules()
                    if isinstance(module, torch.nn.Linear)
                    and (name.startswith('model.visual.') or name == 'lm_head')]
    del skeleton
    if not skip_modules or 'lm_head' not in skip_modules:
        raise RuntimeError('Could not enumerate visual/lm_head quantization exclusions')
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip_modules)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID,
        quantization_config=quant, dtype=torch.float16, device_map={"": 0},
        attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "left"
    bad = [n for n, m in model.named_modules()
           if ("visual" in n or "lm_head" in n) and "4bit" in type(m).__name__.lower()]
    if bad:
        raise RuntimeError(f"Visual/lm_head unexpectedly quantized: {bad[:5]}")
    return model, processor


def apply_setup(model, config):
    import torch
    from peft import LoraConfig, get_peft_model
    targets = LLM_TARGET
    expected = 252
    if config["lora"]["targets"] == "llm+vit":
        targets = f"(?:{LLM_TARGET}|{VIT_TARGET})"
        expected = 348
    matched = [n for n, _ in model.named_modules() if re.fullmatch(targets, n)]
    if len(matched) != expected:
        raise RuntimeError(f"Expected {expected} LoRA targets, found {len(matched)}. Inspect model tree.")
    for p in model.parameters():
        p.requires_grad_(False)
    model.enable_input_require_grads()
    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"], target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    for n, p in model.named_parameters():
        if projector_name(n) and config["train_projector"]:
            p.data = p.data.to(torch.float32)
            p.requires_grad_(True)
        if p.requires_grad:
            # GradScaler must unscale fp32 gradients, including LoRA in fp16 ViT.
            p.data = p.data.to(torch.float32)
            if "lora_" not in n and not (projector_name(n) and config["train_projector"]):
                raise RuntimeError(f"Unexpected trainable parameter: {n}")
    return model, matched


def save_checkpoint(model, processor, directory, run, optimizer=None, scheduler=None, scaler=None):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory / "adapter")
    processor.save_pretrained(directory / "processor")
    extra = {n: p.detach().cpu() for n, p in model.named_parameters()
             if projector_name(n) and p.requires_grad}
    if extra:
        torch.save(extra, directory / "projector.pt")
    if optimizer is not None:
        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(), "step": run["completed_steps"]},
                   directory / "training_state.pt")
    write_json(directory / "run.json", run)


def load_checkpoint(model, directory):
    import torch
    from peft import PeftModel
    directory = Path(directory)
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    model = PeftModel.from_pretrained(model, directory / "adapter", is_trainable=False)
    if run["config"]["train_projector"]:
        saved = torch.load(directory / "projector.pt", map_location="cpu", weights_only=True)
        params = dict(model.named_parameters())
        expected = {n for n in params if projector_name(n)}
        if set(saved) != expected:
            raise ValueError("Saved projector keys do not match loaded model")
        for n, value in saved.items():
            if value.shape != params[n].shape:
                raise ValueError(f"Projector shape mismatch: {n}")
            params[n].data = value.to(device=params[n].device, dtype=torch.float32)
    return model, run


def diagnostic(model, processor, entry, data_root, coord):
    import torch
    from PIL import Image
    with Image.open(Path(data_root) / entry["image_path"]) as im:
        encoded = encode_example(processor, im.convert("RGB"), entry, coord)
    labels = encoded["labels"][0]
    answer = processor.tokenizer.decode(labels[labels != -100], skip_special_tokens=False)
    image_tokens = int((encoded["input_ids"] == model.config.image_token_id).sum())
    if (entry["width"], entry["height"]) == (640, 360) and image_tokens != 220:
        raise ValueError(f"Expected 220 vision tokens for 640x360; found {image_tokens}")
    batch = {k: v.to("cuda") for k, v in encoded.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite diagnostic loss")
    return {"image_id": entry["image_id"], "vision_tokens": image_tokens,
            "supervised_tokens": int((labels != -100).sum()), "decoded_labels": answer,
            "forward_loss": float(loss)}


def train(config, data_root, output, fraction=1.0, coord="norm1000", backup=None):
    import torch
    from functools import partial
    from torch.utils.data import DataLoader
    from transformers import set_seed
    set_seed(config["seed"])
    output = Path(output)
    if (output / "run.json").exists():
        raise FileExistsError(f"Run already exists: {output}; choose a new output directory")
    rows = subset_train(read_jsonl(Path(data_root)/"splits/train.jsonl"), fraction, config["seed"])
    split_meta = json.loads((Path(data_root)/"splits/meta.json").read_text(encoding="utf-8"))
    model, processor = load_base()
    architecture = architecture_counts(model)
    model, matched = apply_setup(model, config)
    processor.tokenizer.padding_side = "right"
    diag = diagnostic(model, processor, rows[0], data_root, coord)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config["lr"], weight_decay=0)
    steps = config["max_steps"]
    warmup = max(1, round(steps*.05))
    def lr_factor(step):
        if step < warmup:
            return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1, (step-warmup)/(steps-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    loader = DataLoader(RoadDataset(rows, data_root, processor, coord),
        batch_size=config["batch_size"], shuffle=True, num_workers=0,
        collate_fn=partial(collate_examples, pad_token_id=processor.tokenizer.pad_token_id))
    iterator = iter(loader)
    run = {"config": config, "coord": coord, "train_fraction": fraction,
        "dataset_version": split_meta["dataset_version"], "train_image_ids": [e["image_id"] for e in rows],
        "trainable_params": sum(p.numel() for p in params), "matched_modules": matched,
        "diagnostic": diag, "architecture": architecture, "environment": environment(), "completed_steps": 0,
        "peak_vram_gb": 0, "train_minutes": 0, "log": []}
    print(json.dumps({'diagnostic': diag, 'trainable_params': run['trainable_params'],
                      'matched_lora_modules': len(matched), 'architecture': architecture},
                     ensure_ascii=False, indent=2), flush=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    step = 1
    consecutive_overflows = 0
    while step <= steps:
        total_loss = 0.0
        for _ in range(config["grad_accum"]):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss / config["grad_accum"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss is NaN/Inf at optimizer step {step}")
            total_loss += float(loss.detach())
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=False)
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < before or not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            consecutive_overflows += 1
            if consecutive_overflows >= 20:
                raise FloatingPointError("20 consecutive gradient overflows; training cannot progress")
            print(f"{config['name']} gradient overflow: scale={scaler.get_scale()}; "
                  f"retry optimizer step {step}", flush=True)
            continue
        consecutive_overflows = 0
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        run["completed_steps"] = step
        run["log"].append({"step": step, "loss": total_loss, "lr": scheduler.get_last_lr()[0]})
        if step == 1 or step % 10 == 0:
            print(f"{config['name']} step {step}/{steps}: loss={total_loss:.4f}", flush=True)
        if step % 100 == 0 or step == steps:
            torch.cuda.synchronize()
            run["peak_vram_gb"] = torch.cuda.max_memory_allocated()/1e9
            run["train_minutes"] = (time.perf_counter()-started)/60
            checkpoint = output / f"step-{step}"
            save_checkpoint(model, processor, checkpoint, run, optimizer, scheduler, scaler)
            write_json(output / "run.json", run)
            if backup:
                shutil.copytree(checkpoint, Path(backup)/output.name/checkpoint.name, dirs_exist_ok=True)
                write_json(Path(backup)/output.name/"run.json", run)
        step += 1
    return run


def predict(data_root, split, path, checkpoint=None, coord="norm1000", batch_size=4,
            max_new_tokens=256, limit=None):
    import torch
    from PIL import Image
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Predictions already exist: {path}. Use a new output name.")
    model, processor = load_base()
    run = None
    if checkpoint:
        model, run = load_checkpoint(model, checkpoint)
        if run["coord"] != coord:
            raise ValueError("Prediction coordinate scale differs from training")
        meta = json.loads((Path(data_root)/"splits/meta.json").read_text(encoding="utf-8"))
        if run["dataset_version"] != meta["dataset_version"]:
            raise ValueError("Dataset version differs from training")
    model.eval()
    model.config.use_cache = True
    processor.tokenizer.padding_side = "left"
    entries = read_jsonl(Path(data_root)/"splits"/f"{split}.jsonl")
    if limit:
        entries = entries[:limit]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    # Preserve partial raw output if the runtime disconnects. No silent resumption.
    with path.open("x", encoding="utf-8") as f:
        for start in range(0, len(entries), batch_size):
            chunk = entries[start:start+batch_size]
            images = []
            for e in chunk:
                with Image.open(Path(data_root)/e["image_path"]) as im:
                    images.append(im.convert("RGB"))
            chats = [[user_message(im, coord)] for im in images]
            inputs = processor.apply_chat_template(chats, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", processor_kwargs={"padding": True})
            inputs.pop("token_type_ids", None)
            inputs = inputs.to("cuda")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            raw = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for entry, text in zip(chunk, raw):
                f.write(json.dumps({"image_id": entry["image_id"], "raw": text}, ensure_ascii=False)+"\n")
            f.flush()
            print(f"{split}: {min(start+batch_size, len(entries))}/{len(entries)}", flush=True)
    torch.cuda.synchronize()
    summary = {"split": split, "coord": coord, "n_images": len(entries),
        "checkpoint": str(checkpoint) if checkpoint else None, "max_new_tokens": max_new_tokens,
        "inference_minutes": (time.perf_counter()-started)/60,
        "peak_vram_gb": torch.cuda.max_memory_allocated()/1e9,
        "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "environment": environment()}
    write_json(path.with_suffix(".run.json"), summary)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return summary
