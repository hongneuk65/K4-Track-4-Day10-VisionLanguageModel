"""Build a factual report only after all measured experiments are complete."""
import argparse
import json
import statistics
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import evaluate, parse_output
from lab.data import audit_data


def build(author, root='.'):
    root = Path(root)
    def load(path):
        return json.loads((root / path).read_text(encoding='utf-8'))
    names = ('e0', 'e1', 'e2', 'e3', 'e4_data_size_0.1', 'e4_data_size_1.0')
    audit = audit_data(root / 'data')
    ev = {n: load(f'report/evidence/{n}.json') for n in names}
    runs = {n: load(f'work/runs/{n}/run.json') for n in ('e1', 'e2', 'e3', 'e4_data_size_0.1')}
    val = {n: load(f'work/validation/{n}.json') for n in ('e1', 'e2', 'e3')}
    best = max(val, key=lambda n: val[n]['micro']['f1'])
    for n, e in ev.items():
        if e['split'] != 'test' or e['missing_predictions'] or e['dataset_version'] != audit['dataset_version']:
            raise ValueError(f'Incomplete or inconsistent evidence: {n}')
    for n, r in runs.items():
        if r['completed_steps'] != 300 or r['dataset_version'] != audit['dataset_version']:
            raise ValueError(f'Incomplete run: {n}')
        if any(r[k] <= 0 for k in ('trainable_params', 'peak_vram_gb', 'train_minutes')):
            raise ValueError(f'Missing measured resources: {n}')
    base_cfg = {k: v for k, v in runs[best]['config'].items() if k != 'name'}
    small_cfg = {k: v for k, v in runs['e4_data_size_0.1']['config'].items() if k != 'name'}
    if base_cfg != small_cfg or runs['e4_data_size_0.1']['train_fraction'] != 0.1:
        raise ValueError('E4 must change only data size')
    test = read_jsonl(root / 'data/splits/test.jsonl')
    raw_e0 = read_jsonl(root / 'work/e0.jsonl')
    wrong_scale = evaluate(test, {r['image_id']: r['raw'] for r in raw_e0}, coord='pixel')['micro']['f1']
    malformed = [r for r in raw_e0 if parse_output(r['raw']) is None]
    repeated = [r for r in raw_e0 if r['raw'].count('"bbox_2d"') > r['raw'].count('{')]
    captions = load('report/figures/predictions/captions.json')
    if len(captions) < 6 or any(not (root / c['file']).is_file() for c in captions):
        raise ValueError('Six prediction figures required')
    f = lambda x: f'{x:.4f}'
    lines = [f'Họ tên: {author}', f'`dataset_version`: `{audit["dataset_version"]}`', '',
        '## Bảng tổng hợp', '',
        '| Mã | Tham số train | VRAM đỉnh GB | Phút train | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |',
        '|---|---:|---:|---:|---:|---|---:|---:|---:|---:|']
    for n in names:
        e, m = ev[n], ev[n]['meta']
        cells = [n, str(m['trainable_params']), f(m['peak_vram_gb']), f(m['train_minutes']),
            f(e['micro']['f1']), f'{f(e["f1_ci95"][0])}–{f(e["f1_ci95"][1])}', f(e['fp_per_image'])]
        cells += [f(e['per_class'][c]['f1']) for c in CLASSES]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', f'E0 không train; thời gian inference {f(ev["e0"]["meta"]["inference_minutes"])} phút. '
        'VRAM E0 là đỉnh inference; VRAM E1–E4 là đỉnh train. Không cộng thời gian validation/test vào phút train.', '',
        '### Câu 1 — Kiến trúc',
        'Số tham số logic của base model được đếm trước khi gắn LoRA (không dùng số byte lưu trữ 4-bit):',
        '```json', json.dumps(runs['e1']['architecture'], ensure_ascii=False, indent=2), '```',
        'E1 cập nhật LoRA LLM; E2 cập nhật thêm toàn bộ merger và deepstack mergers; E3 cập nhật thêm LoRA ViT.',
        '; '.join(f'{n.upper()}: {runs[n]["trainable_params"]:,} tham số train, '
                  f'{len(runs[n]["matched_modules"])} module LoRA' for n in ('e1', 'e2', 'e3')), '',
        '### Câu 2 — E0',
        f'parse_valid_rate = {f(ev["e0"]["parse_valid_rate"])}. Có {len(malformed)} output không qua parser. '
        f'Có {len(repeated)} output chứa nhiều khóa bbox_2d hơn số object mở, một dấu hiệu lặp khóa. '
        'Output dài có thể bị cắt ở 256 token; lặp khóa làm JSON có thể hợp lệ cú pháp nhưng không biểu diễn đủ các object.',
        'Ví dụ thô trên test (giữ nguyên, không sửa để chấm):', '```text',
        (malformed or repeated or raw_e0)[0]['raw'], '```', '',
        '### Câu 3 — Thang tọa độ',
        'YOLO [0.5,0.5,0.2,0.2] trên ảnh 640×360 thành pixel [256,144,384,216] hoặc norm1000 [400,400,600,600]. '
        'Kiểm tra đơn vị tổng hợp cho F1=1 nếu chấm đúng pixel và F1=0 nếu hiểu nhầm là norm1000.',
        f'Chấm lại cùng raw E0 theo pixel (không chạy lại model) được F1={f(wrong_scale)}, '
        f'trong khi thang đúng norm1000 là {f(ev["e0"]["micro"]["f1"])}. '
        'Đây chỉ là kiểm chứng thang đo; evidence nộp dùng norm1000.', '',
        '### Câu 4 — Mask loss',
        'Prompt, token ảnh và padding có labels=-100. Chỉ đáp án assistant và token kết thúc được supervise. '
        'Nếu supervise prompt, model phải học tái tạo yêu cầu và các placeholder ảnh, làm lệch mục tiêu học JSON định vị.',
        'Chẩn đoán forward trước train E1:', '```json',
        json.dumps(runs['e1']['diagnostic'], ensure_ascii=False, indent=2), '```', '',
        '### Câu 5 — Projector',
        'Lượng tử hóa 4-bit áp dụng lên LLM; từng lớp Linear của visual và lm_head được bỏ qua bằng tên đầy đủ. '
        'Trọng số Linear4bit nền không được full finetune như tham số float thông thường. '
        'Merger và deepstack mergers được đưa về float32 khi train E2/E3 để giữ tham số cập nhật ổn định; '
        'forward dùng autocast fp16 và GradScaler trên T4. ViT nền vẫn đóng băng; E3 mở LoRA ViT.',
        'Adapter PEFT và projector.pt được lưu riêng. Khi nạp projector kiểm tra toàn bộ key và shape, '
        'tránh vô tình chấm E2/E3 bằng projector gốc.', '',
        '### Câu 6 — Split theo phiên quay',
        'Frame gần nhau của một video tương tự nhau; chia ngẫu nhiên làm model được kiểm tra trên cảnh đã gần như thấy trong train. '
        'Audit xác nhận không giao image_id hay session giữa các split.',
        '```json', json.dumps(audit['counts'], ensure_ascii=False, indent=2), '```', '',
        '### Câu 7 — So sánh E1/E2/E3']
    for n in ('e2', 'e3'):
        lines.append(f'{n.upper()} so với E1: ΔF1={ev[n]["micro"]["f1"]-ev["e1"]["micro"]["f1"]:+.4f}; '
            f'ΔVRAM={runs[n]["peak_vram_gb"]-runs["e1"]["peak_vram_gb"]:+.4f} GB; '
            f'Δthời gian={runs[n]["train_minutes"]-runs["e1"]["train_minutes"]:+.4f} phút.')
    lines += [f'Chọn {best.upper()} bằng F1 validation: ' + ', '.join(
        f'{n.upper()}={f(val[n]["micro"]["f1"])}' for n in val) + '. '
        'Việc mở projector/ViT tăng số tham số và bộ nhớ optimizer; cần cân nhắc chênh lệch F1 cùng chi phí trong bảng. '
        'Chưa có kiểm định paired bootstrap để khẳng định khác biệt có ý nghĩa thống kê.', '',
        '### Câu 8 — Khoảng tin cậy',
        'Test có 311 ảnh từ 3 phiên. Độ rộng CI95: ' + ', '.join(
            f'{n.upper()}={f(ev[n]["f1_ci95"][1]-ev[n]["f1_ci95"][0])}' for n in names) + '.',
        'Scorer bootstrap theo ảnh 1000 lần. Frame trong cùng phiên tương quan nên CI theo ảnh có thể hẹp hơn bất định giữa các phiên. '
        'CI chồng lấn không chứng minh tương đương; không chọn setup hay điều chỉnh siêu tham số theo test.', '',
        '### Câu 9 — Phân tích lỗi theo lớp']
    weak = min(CLASSES, key=lambda c: ev[best]['per_class'][c]['f1'])
    lines.append(f'Lớp yếu nhất của {best.upper()} trên test là {weak}, F1={f(ev[best]["per_class"][weak]["f1"])}.')
    for c in CLASSES:
        boxes = [o['yolo'] for row in test for o in row['objects'] if o['label'] == c]
        lines.append(f'{c}: {len(boxes)} box test; diện tích trung vị '
                     f'{(statistics.median(b[2]*b[3] for b in boxes) if boxes else 0)*100:.3f}% ảnh. '
                     f'F1={f(ev[best]["per_class"][c]["f1"])}.')
    lines += ['Các số này mô tả kích thước và số mẫu, chưa chứng minh nguyên nhân nhân quả. '
        'Box mảnh/nhỏ nhạy với sai lệch vài pixel ở ngưỡng IoU 0.5. '
        'Ảnh dưới cho biết lỗi bỏ sót (FN), box thừa (FP) và số khớp đúng (TP), không chỉ tính hợp lệ JSON.', '',
        '### Câu 10 — E4',
        f'Biến thay đổi là tỷ lệ ảnh train: 10% ({len(runs["e4_data_size_0.1"]["train_image_ids"])} ảnh) '
        f'và 100% ({len(runs[best]["train_image_ids"])} ảnh), từ setup {best.upper()} chọn trên validation. '
        'Giữ nguyên seed, rank, alpha, dropout, LR, lịch LR, 300 bước, batch, prompt và norm1000. '
        'Giá trị 100% dùng lại run gốc.',
        f'F1 10%={f(ev["e4_data_size_0.1"]["micro"]["f1"])}; '
        f'100%={f(ev["e4_data_size_1.0"]["micro"]["f1"])}; '
        f'Δ(10%−100%)={ev["e4_data_size_0.1"]["micro"]["f1"]-ev["e4_data_size_1.0"]["micro"]["f1"]:+.4f}. '
        'Cùng 300 bước khiến tập 10% được lặp lại nhiều hơn. Đây là một seed, một subset; '
        'cần thêm seed và split độc lập trước khi khái quát kết luận.', '', '## Ảnh minh họa']
    for c in captions:
        lines += ['', f'![{c["image_id"]}]({Path(c["file"]).as_posix()})',
            f'{c["image_id"]}: TP={c["tp"]}, FP={c["fp"]}, FN={c["fn"]}; '
            f'parser hợp lệ={c["parse_valid"]}. Box xanh là GT, đỏ là dự đoán. '
            + ('Không có box thừa/bỏ sót ở ngưỡng IoU 0.5.' if not c['fp'] and not c['fn'] else
               f'Có {c["fn"]} box GT không khớp và {c["fp"]} box dự đoán không khớp cùng lớp ở IoU 0.5.')]
    lines += ['', '## Nhìn lại',
        'Đã gặp nhãn crack chiều cao 0 trong vlcsnap-00058. Đáp án train mở trục co thành 1 đơn vị nguyên; '
        'giữ nguyên dataset split và GT của scorer. Box diện tích 0 không thể đạt IoU 0.5 với box diện tích dương.',
        'Transformers 5.19 yêu cầu processor_kwargs để pad inference batch và danh sách module đầy đủ để bỏ qua quantization visual. '
        'Các phiên Colab là tạm; cần tải ZIP evidence và giữ riêng adapter/projector trước khi ngắt runtime.',
        f'Kiểm tra tiêu chí E1: parse_valid_rate={f(ev["e1"]["parse_valid_rate"])} '
        f'({"đạt" if ev["e1"]["parse_valid_rate"] >= .95 else "chưa đạt"} ngưỡng 0.95); '
        f'ΔF1 so E0={ev["e1"]["micro"]["f1"]-ev["e0"]["micro"]["f1"]:+.4f} '
        f'({"đạt" if ev["e1"]["micro"]["f1"]-ev["e0"]["micro"]["f1"] >= .1 else "chưa đạt"} ngưỡng 0.10).',
        'E5: chỉ triển khai và kiểm thử reward; chưa chạy GRPO, không tuyên bố kết quả bonus. '
        'Giới hạn chưa giải quyết: độ ổn định giữa seed, CI theo session và nguyên nhân hình ảnh cụ thể cần thêm kiểm chứng.']
    destination = root / 'REPORT.md'
    destination.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--author', required=True)
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    print(build(args.author, args.root))
