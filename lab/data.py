"""Coordinates, chat encoding and multimodal collation for road grounding."""
import json
import random
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl


def prompt_for(coord="norm1000"):
    units = ("integer coordinates normalized to 0-1000 along each image axis"
             if coord == "norm1000" else "integer pixel coordinates in the original image")
    return ("Detect all potholes, cracks and manholes in this road image. "
            "Use only these labels: pothole, crack, manhole. "
            "Return ONLY a JSON list, with no explanation or markdown. "
            'Each item must have the form {"label":"pothole","bbox_2d":[x1,y1,x2,y2]}. '
            f"Use {units}; x1 < x2 and y1 < y2. "
            "Order objects from top to bottom, then left to right. Return [] if none.")


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    if label not in CLASSES or coord not in ("norm1000", "pixel"):
        raise ValueError("Unknown label or coordinate scale")
    xc, yc, w, h = yolo
    if w < 0 or h < 0:
        raise ValueError(f"Negative source box dimensions: {yolo}")
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    values = ((xc-w/2)*sx, (yc-h/2)*sy, (xc+w/2)*sx, (yc+h/2)*sy)
    box = [max(0, min(limit, round(v))) for v, limit in zip(values, (sx, sy, sx, sy))]
    # Tiny boxes (including a zero-height annotation in the supplied dataset)
    # can collapse after integer rounding. Preserve the training label
    # by expanding only the collapsed axis by one unit, within the coordinate range.
    for start, end, limit in ((0, 2, sx), (1, 3, sy)):
        if box[start] == box[end]:
            if box[end] < limit:
                box[end] += 1
            else:
                box[start] -= 1
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Invalid box after coordinate conversion: {box}")
    return {"label": label, "bbox_2d": box}


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    boxes.sort(key=lambda o: (o["bbox_2d"][1], o["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def user_message(image, coord):
    return {"role": "user", "content": [{"type": "image", "image": image},
            {"type": "text", "text": prompt_for(coord)}]}


def encode_example(processor, image, entry, coord="norm1000"):
    import torch
    user = user_message(image, coord)
    answer = objects_to_answer(entry["objects"], coord, entry["width"], entry["height"])
    opts = dict(tokenize=True, return_dict=True, return_tensors="pt")
    prefix = processor.apply_chat_template([user], add_generation_prompt=True, **opts)
    full = processor.apply_chat_template([user, {"role": "assistant", "content": answer}],
                                         add_generation_prompt=False, **opts)
    n = prefix["input_ids"].shape[1]
    if not torch.equal(full["input_ids"][:, :n], prefix["input_ids"]):
        raise ValueError("Chat template prompt is not an exact prefix of the full sequence")
    full.pop("token_type_ids", None)
    full["labels"] = full["input_ids"].clone()
    full["labels"][:, :n] = -100
    if not (full["labels"] != -100).any():
        raise ValueError("No supervised answer tokens")
    return dict(full)


def collate_examples(examples, pad_token_id):
    import torch
    length = max(e["input_ids"].shape[1] for e in examples)
    output = {}
    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "labels"):
        if key not in examples[0]:
            continue
        value = -100 if key == "labels" else pad_token_id if key == "input_ids" else 0
        output[key] = torch.cat([torch.nn.functional.pad(e[key],
                                (0, length-e[key].shape[1]), value=value) for e in examples])
    for key in ("pixel_values", "image_grid_thw"):
        if key in examples[0]:
            output[key] = torch.cat([e[key] for e in examples], dim=0)
    return output


class RoadDataset:
    def __init__(self, entries, root, processor, coord):
        self.entries, self.root, self.processor, self.coord = entries, Path(root), processor, coord

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        from PIL import Image
        entry = self.entries[index]
        with Image.open(self.root / entry["image_path"]) as im:
            image = im.convert("RGB")
        return encode_example(self.processor, image, entry, self.coord)


def subset_train(entries, fraction, seed):
    if not 0 < fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    indices = list(range(len(entries)))
    random.Random(seed).shuffle(indices)
    return [entries[i] for i in indices[:max(1, round(len(entries)*fraction))]]


def audit_data(root="data"):
    root = Path(root)
    rows = {s: read_jsonl(root / "splits" / f"{s}.jsonl") for s in ("train", "val", "test")}
    for i, a in enumerate(rows):
        for b in list(rows)[i+1:]:
            for key in ("image_id", "session"):
                if {r[key] for r in rows[a]} & {r[key] for r in rows[b]}:
                    raise ValueError(f"Leakage in {key}: {a}/{b}")
    counts = {}
    degenerate_annotations = []
    for split, entries in rows.items():
        if len({e["image_id"] for e in entries}) != len(entries):
            raise ValueError(f"Duplicate image ID in {split}")
        boxes = {c: 0 for c in CLASSES}
        for e in entries:
            if not (root / e["image_path"]).is_file():
                raise FileNotFoundError(e["image_path"])
            objects_to_answer(e["objects"], width=e["width"], height=e["height"])
            for o in e["objects"]:
                boxes[o["label"]] += 1
                if o['yolo'][2] == 0 or o['yolo'][3] == 0:
                    degenerate_annotations.append({'split': split, 'image_id': e['image_id'],
                                                   'label': o['label'], 'yolo': o['yolo']})
        counts[split] = {"images": len(entries), "boxes": boxes,
                         "sessions": len({e["session"] for e in entries})}
    meta = json.loads((root / "splits" / "meta.json").read_text(encoding="utf-8"))
    for s, c in counts.items():
        if c["images"] != meta["counts"][s]["images"] or c["boxes"] != meta["counts"][s]["boxes"]:
            raise ValueError("Counts do not match split metadata")
    return {"dataset_version": meta["dataset_version"], "counts": counts,
            "degenerate_annotations": degenerate_annotations}
