"""Optional E5 prerequisite: pure IoU reward, without a GRPO run claim."""
from scripts.score import parse_output, iou_xyxy


def iou_reward(completion, gt_boxes):
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not pred and not gt_boxes:
        return 1.0
    pairs = sorted(((iou_xyxy(g['bbox'], p['bbox']), i, j)
                    for i, g in enumerate(gt_boxes) for j, p in enumerate(pred)
                    if g['label'] == p['label']), reverse=True)
    used_g, used_p, total = set(), set(), 0.0
    for value, i, j in pairs:
        if value < .5:
            break
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            total += value
    return .2 + .8*total/max(len(gt_boxes), len(pred), 1)
