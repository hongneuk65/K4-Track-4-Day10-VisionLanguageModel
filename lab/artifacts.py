"""Generate visual evidence and a report draft from measured outputs only."""
import json
from pathlib import Path

from lab.data import audit_data
from lab.runtime import write_json
from scripts.labcommon import read_jsonl, CLASSES
from scripts.score import _gt_boxes, _match, _to_norm1000, parse_output, evaluate


def illustrate(data_root, pred_path=None, split="val", coord="norm1000", out="report/figures", count=6):
    from PIL import Image, ImageDraw
    entries = read_jsonl(Path(data_root)/"splits"/f"{split}.jsonl")
    predictions = {r["image_id"]: r["raw"] for r in read_jsonl(pred_path)} if pred_path else None
    candidates = []
    for entry in entries:
        if predictions is not None and entry["image_id"] not in predictions:
            continue
        gt = _gt_boxes(entry)
        raw = predictions[entry["image_id"]] if predictions is not None else None
        parsed = parse_output(raw)
        pred = _to_norm1000(parsed or [], coord, entry["width"], entry["height"])
        per, _ = _match(gt, pred)
        tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
        category = "correct" if not fp and not fn else "mixed" if tp else "error"
        candidates.append((category, entry, gt, pred, tp, fp, fn, parsed is not None))
    chosen = []
    # Round-robin selects successes and failures when both exist.
    groups = [[c for c in candidates if c[0] == category] for category in ("correct", "mixed", "error")]
    while len(chosen) < count and any(groups):
        for group in groups:
            if group and len(chosen) < count:
                chosen.append(group.pop(0))
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    captions = []
    for category, entry, gt, pred, tp, fp, fn, valid in chosen:
        with Image.open(Path(data_root)/entry["image_path"]) as im:
            image = im.convert("RGB")
        canvas = Image.new("RGB", (image.width, image.height+52), "white")
        canvas.paste(image)
        draw = ImageDraw.Draw(canvas)
        for boxes, color, tag in ((gt, "lime", "GT"), (pred, "red", "PRED")):
            for b in boxes:
                x1, y1, x2, y2 = b["bbox"]
                xy = (x1/1000*image.width, y1/1000*image.height,
                      x2/1000*image.width, y2/1000*image.height)
                # Keep both outlines visible even for a perfectly matching prediction.
                draw.rectangle(xy, outline=color, width=3 if tag == "GT" else 1)
                label_y = max(0, xy[1]-12) if tag == "GT" else min(image.height-12, xy[3]+2)
                draw.text((xy[0], label_y), f"{tag}: {b['label']}", fill=color)
        title = (f"{entry['image_id']} | TP={tp} FP={fp} FN={fn} | parse={valid}"
                 if predictions is not None else f"{entry['image_id']} | Ground truth labels")
        draw.text((6, image.height+6), title, fill="black")
        draw.text((6, image.height+27), "GT = green; prediction = red; matching at IoU >= 0.5", fill="black")
        file = output/f"{entry['image_id']}.png"
        canvas.save(file)
        captions.append({"image_id": entry["image_id"], "file": str(file), "category": category,
                         "tp": tp, "fp": fp, "fn": fn, "parse_valid": valid})
    write_json(output/"captions.json", captions)
    return captions


def scale_probe():
    entry = {"image_id": "controlled", "width": 640, "height": 360,
             "objects": [{"label": "pothole", "yolo": [.5, .5, .2, .2]}]}
    raw = '[{"label":"pothole","bbox_2d":[256,144,384,216]}]'
    return {"description": "Synthetic unit-scale check, not a model test result",
            "pixel_f1": evaluate([entry], {"controlled": raw}, coord="pixel")["micro"]["f1"],
            "wrong_norm1000_f1": evaluate([entry], {"controlled": raw})["micro"]["f1"]}


def report_draft(data_root, runs_root, evidence_root="report/evidence", out="report/REPORT_DRAFT.md"):
    evidence_root, runs_root = Path(evidence_root), Path(runs_root)
    names = ["e0", "e1", "e2", "e3", "e4_data_size_0.1", "e4_data_size_1.0"]
    evidence = {n: json.loads((evidence_root/f"{n}.json").read_text(encoding="utf-8")) for n in names}
    audit = audit_data(data_root)
    for name, e in evidence.items():
        if e["split"] != "test" or e["missing_predictions"] or e["dataset_version"] != audit["dataset_version"]:
            raise ValueError(f"Incomplete/inconsistent evidence: {name}")
    runs = {n: json.loads((runs_root/n/"run.json").read_text(encoding="utf-8")) for n in ("e1", "e2", "e3")}
    lines = ["# Báo cáo Lab Day 10 — bản nháp từ kết quả thực tế", "",
        "Họ tên: <điền họ tên>", f"dataset_version: `{audit['dataset_version']}`", "",
        "Bản nháp này lấy số từ evidence. Cần đọc output thô và ảnh để hoàn thiện phân tích cá nhân.", "",
        "## Bảng tổng hợp", "",
        "| Mã | Tham số train | VRAM GB | Phút train | F1 | CI95 | FP/ảnh | pothole | crack | manhole |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for name, e in evidence.items():
        meta = e["meta"]
        values = [name, str(meta.get("trainable_params", 0)), str(meta.get("peak_vram_gb", "—")),
                  str(meta.get("train_minutes", "—")), f"{e['micro']['f1']:.4f}",
                  f"[{e['f1_ci95'][0]:.4f}, {e['f1_ci95'][1]:.4f}]", f"{e['fp_per_image']:.4f}"]
        values += [f"{e['per_class'][c]['f1']:.4f}" for c in CLASSES]
        lines.append("| "+" | ".join(values)+" |")
    lines += ["", f"E0 không train; thời gian suy luận: {evidence['e0']['meta'].get('inference_minutes', 'xem inference log')} phút.",
              "", "### Câu 1 — Kiến trúc", "E1 cập nhật LoRA LLM; E2 thêm projector; E3 thêm LoRA ViT.",
              "Số tham số train đo được: "+", ".join(f"{n.upper()}: {r['trainable_params']:,}" for n, r in runs.items()),
              "Số tham số logic từng khối (không dùng kích thước lưu trữ 4-bit):",
              "```json", json.dumps(runs['e1']['architecture'], indent=2), "```", "",
              "### Câu 2 — E0", f"parse_valid_rate = {evidence['e0']['parse_valid_rate']:.4f}.",
              "<điền hai lỗi phổ biến đã quan sát và một output thô>", "",
              "### Câu 3 — Thang tọa độ", "Ví dụ kiểm soát YOLO [0.5,0.5,0.2,0.2] trên ảnh 640×360:",
              "pixel [256,144,384,216]; norm1000 [400,400,600,600]. Chấm pixel đúng F1=1; nhầm sang norm1000 F1=0.",
              "Đây là kiểm tra đơn vị bằng dữ liệu tổng hợp, không phải số đo F1 của model.", "",
              "### Câu 4 — Mask loss", "Chỉ supervise đáp án assistant và token kết thúc; prompt và padding có labels=-100.",
              "Nếu supervise cả prompt, mục tiêu loss thêm phần lặp yêu cầu đầu vào và các token ảnh, làm lệch mục tiêu học JSON định vị.",
              f"Chẩn đoán E1: {runs['e1']['diagnostic']['vision_tokens']} token ảnh; "
              f"{runs['e1']['diagnostic']['supervised_tokens']} token supervise.", "",
              "### Câu 5 — Projector", "visual và lm_head được bỏ qua khi lượng tử hóa. Projector E2/E3 và các tham số train ở float32, forward dùng autocast fp16.",
              "Projector được lưu riêng trong projector.pt và kiểm tra toàn bộ key/shape khi nạp; trọng số nền 4-bit không được full finetune theo cách này.", "",
              "### Câu 6 — Split theo phiên", "Không session hoặc image_id nào giao nhau giữa các split theo audit.",
              "Frame gần nhau trong cùng video gây rò rỉ nếu chia ngẫu nhiên. Số đếm thực tế:",
              "```json", json.dumps(audit["counts"], ensure_ascii=False, indent=2), "```", "",
              "### Câu 7 — So sánh", "<điền phân tích F1, CI95, VRAM và thời gian từ bảng trên>", "",
              "### Câu 8 — Khoảng tin cậy", "Độ rộng CI95: "+", ".join(
                  f"{n}: {e['f1_ci95'][1]-e['f1_ci95'][0]:.4f}" for n, e in evidence.items()),
              "Scorer bootstrap 1000 lần theo ảnh. Tương quan giữa frame cùng phiên có thể khiến CI chưa phản ánh đầy đủ bất định.",
              "CI chồng lấn không tự động chứng minh tương đương; không kết luận chắc chắn từ chênh lệch nhỏ.", "",
              "### Câu 9 — Phân tích theo lớp", "<điền nguyên nhân lớp yếu dựa trên ảnh, không chỉ suy đoán từ F1>", "",
              "### Câu 10 — E4", "Đổi tỷ lệ train 10%/100%; cùng seed, LR, số bước, batch, rank, prompt và thang tọa độ.",
              "Giữ 300 bước khiến subset nhỏ được lặp lại nhiều hơn. <điền kết luận từ hai giá trị F1 và CI95>", "",
              "## Ảnh minh họa", ""]
    captions_file = Path("report/figures/predictions/captions.json")
    if captions_file.exists():
        for item in json.loads(captions_file.read_text(encoding="utf-8")):
            relative = Path(item["file"]).as_posix().removeprefix("report/")
            lines += [f"![{item['image_id']}]({relative})", f"TP={item['tp']}, FP={item['fp']}, FN={item['fn']}. <điền nhận xét ảnh>", ""]
    lines += ["## Nhìn lại",
              "Nhãn suy biến trong dataset gốc (không sửa ground truth của scorer):",
              "```json", json.dumps(audit.get('degenerate_annotations', []), ensure_ascii=False, indent=2), "```",
              "Đáp án train mở trục co về một đơn vị nguyên để tạo box hợp lệ; box GT diện tích 0 không thể đạt IoU 0.5 với dự đoán diện tích dương.",
              "<điền các giới hạn, sự cố và điều chưa hiểu còn lại>"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines)+"\n", encoding="utf-8")
    return str(out)
