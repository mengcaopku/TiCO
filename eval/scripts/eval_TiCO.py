
import logging
from tqdm import tqdm
from dist_utils import get_rank, get_world_size, init_distributed_mode, CustomDataset_JSONL
import argparse
import torch
import torch.distributed as dist
import transformers
from transformers import AutoProcessor
from PIL import Image, ImageDraw
import json
import re
from collections import defaultdict
import os
import argparse
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoProcessor
from tico.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from tico.inference import inference

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--min_pixels", type=int, default=3136, required=True)
    parser.add_argument("--max_pixels", type=int, default=401408, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--out_img_dir", type=str, required=True)

    return parser.parse_args()

args = parse_args()

def extract_answer(text: str) -> str:
    match = re.search(r"(.*?)<\|im_end\|>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_answer_videor1(text):
    pattern = r'<answer>\s*(.*?)\s*</answer>'
    pattern2 = r'<answer>(.*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        match2 = re.search(pattern2, text, re.DOTALL)
        if match2:
            return match2.group(1).strip()
        else:
            # return ""
            return text.strip()

def parse_yesorno_ans(pred_ans):

    pred_ans = extract_answer_videor1(pred_ans).lower()
    if pred_ans in ["yes", "no"]:
        return pred_ans
    prefix_pred_ans = pred_ans[:4]
    if "yes" in prefix_pred_ans:
        return "yes"
    elif "no" in prefix_pred_ans:
        return "no"
    else:
        return "other"
    
def load_model_and_dataset(model_path, rank, world_size, args):

    eval_dataset = args.eval_dataset
    min_pixels = args.min_pixels
    max_pixels = args.max_pixels
    model_name_or_path = model_path

    processor = AutoProcessor.from_pretrained(
        model_path, min_pixels=min_pixels, max_pixels=max_pixels
    )
    tokenizer = processor.tokenizer
    model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="flash_attention_2"
    ).eval()

    dataset = CustomDataset_JSONL(eval_dataset)
    dataset.set_rank_and_world_size(rank, world_size)

    return model, processor, tokenizer, dataset

def bbox_adjust(bbox, input_width, input_height, width, height, min_size=28):
    x1, y1, x2, y2 = bbox

    x1 = min(x1, input_width)
    x2 = min(x2, input_width)
    y1 = min(y1, input_height)
    y2 = min(y2, input_height)
    
    y1 = int(y1/input_height * height)
    x1 = int(x1/input_width * width)
    y2 = int(y2/input_height * height)
    x2 = int(x2/input_width * width)

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    if bbox_width < min_size:
        s = (min_size - bbox_width) // 2
        x1 -= s
        x2 += s
        if x2 - x1 < min_size:
            x2 += 1
    
    if bbox_height < min_size:
        s = (min_size - bbox_height) // 2
        y1 -= s
        y2 += s
        if y2 - y1 < min_size:
            y2 += 1

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    if max(bbox_width, bbox_height) / min(bbox_width, bbox_height) >= 200:
        return []

    return [x1, y1, x2, y2]


def resize_image(image, resize_to_pixels=args.max_pixels):
    image_width, image_height = image.size
    if (resize_to_pixels is not None) and ((image_width * image_height) != resize_to_pixels):
        resize_ratio = (resize_to_pixels / (image_width * image_height)) ** 0.5
        image_width_resized, image_height_resized = int(image_width * resize_ratio), int(image_height * resize_ratio)
        image = image.resize((image_width_resized, image_height_resized))
    return image

def draw_pred_bbox(image: Image.Image, w: int, h: int, bboxes: list,
                   color=(255, 0, 0, 180), width=3, show_index=True):
    overlay = Image.new('RGBA', image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for idx, (x0, y0, x1, y1) in enumerate(bboxes):

        x0_pix, y0_pix = x0 * w, y0 * h
        x1_pix, y1_pix = x1 * w, y1 * h

        draw.rectangle([(x0_pix, y0_pix), (x1_pix, y1_pix)], outline=color, width=width)

        if show_index:
            text_pos = (x0_pix + 4, y0_pix + 4)
            draw.text(text_pos, str(idx), fill=color)

    image = image.convert('RGBA')
    combined = Image.alpha_composite(image, overlay)
    combined = combined.convert('RGB')
    return combined

@torch.inference_mode()
def get_attn_map(image, attn_scores, n_width, n_height, item):
    w, h = image.size
    scores = np.array(attn_scores[0]).reshape(n_height, n_width)
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    score_map = Image.fromarray((scores_norm * 255).astype(np.uint8)).resize((w, h), resample=Image.NEAREST)

    colormap = plt.get_cmap('jet')
    colored_score_map = colormap(np.array(score_map) / 255.0)
    colored_score_map = (colored_score_map[:, :, :3] * 255).astype(np.uint8)
    colored_overlay = Image.fromarray(colored_score_map)

    blended = Image.blend(image, colored_overlay, alpha=0.7)

    blended_draw = ImageDraw.Draw(blended)
    blended = blended.convert('RGB')
    return blended

def _save_pred_products(
    base_name: str,
    image: Image.Image,
    pred: dict,
    output_dir: str,
    prefix: str = "",
    save_image: bool = True,
):

    os.makedirs(output_dir, exist_ok=True)
    w, h = image.size

    topk_points = pred.get("topk_points", [])
    pred_boxes = pred.get("bboxes", [(0.4,0.4,0.5,0.5)])
    if not topk_points:
        topk_points = [(0.5, 0.5)]

    img_with_bbox = draw_pred_bbox(image, w, h, pred_boxes)

    x0, y0, x1, y1 = pred_boxes[0]
    left, top, right, bottom = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    left, top = max(0, left), max(0, top)
    right, bottom = min(w, right), min(h, bottom)
    img_crop_bbox = image.crop((left, top, right, bottom))

    n_width, n_height = pred.get("n_width", None), pred.get("n_height", None)
    attn_scores = pred.get("attn_scores", None)
    att_map = None
    if attn_scores is not None and n_width is not None and n_height is not None:
        att_map = get_attn_map(image, attn_scores, n_width, n_height, item=None)

    bbox_image_path  = os.path.join(output_dir, f"{base_name}_{prefix}with_bbox.png")
    attn_image_path  = os.path.join(output_dir, f"{base_name}_{prefix}attention.png")
    crop_image_path  = os.path.join(output_dir, f"{base_name}_{prefix}_crop.png")

    if save_image:
        img_with_bbox.save(bbox_image_path)
        if att_map is not None:
            att_map.save(attn_image_path)
        else:
            print("No Grounding Token")
        img_crop_bbox.save(crop_image_path)

    return img_crop_bbox

def run(model_path, dataset_path, rank, args, world_size):
    if rank != 0:
        transformers.utils.logging.set_verbosity_error()
        logger.setLevel(transformers.logging.ERROR)

    logger.info(f'loading model and constructing dataset to gpu {rank}...')
    model, processor, tokenizer, dataset = load_model_and_dataset(model_path, rank, world_size, args)

    output_data = []
    done_count = 0
    device = f"cuda:{rank}"

    if rank == 0:
        tbar = tqdm(total=len(dataset))

    for item in dataset:

        problem = item['problem']
        # question_id = item['Question_id']
        inst_locate = problem + " Your task is to locate the region that corresponds to the question. To indicate the region, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: <your_special_token_here>."
        inst_answer = problem + " Refer to the region extracted in the previous turn and provide the final answer."

        ground_truth = item["solution"]
        image_path = os.path.join(dataset_path, item["Image"])

        min_pixels = args.min_pixels
        max_pixels = args.max_pixels
        out_img_dir = args.out_img_dir

        origin_image = Image.open(image_path).convert('RGB')
        w_raw, h_raw = origin_image.size
        if w_raw * h_raw > max_pixels:
            image = resize_image(origin_image)
            w, h = image.size
        else:
            image = origin_image
            w, h = origin_image.size


        generation_list = []
        conv1 = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": inst_locate}]},
        ]

        pred1 = inference(conv1, model, tokenizer, processor, use_placeholder=False, topk=3, conv_iteration=1)
        generation_list.append(pred1["output_text"])

        img_crop_bbox = _save_pred_products(
            base_name="question_id",
            # base_name=question_id,
            image=origin_image,
            pred=pred1,
            output_dir=out_img_dir,
            prefix="",
            save_image=False
        )
        w_crop, h_crop = img_crop_bbox.size
        if w_crop * h_crop > max_pixels:
            img_crop_bbox = resize_image(img_crop_bbox)

        pointer_text = pred1.get("output_text", "<|Ground|>")

        conv2_with_history = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": inst_locate},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<|Ground|>"}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_crop_bbox},
                    {"type": "text", "text": inst_answer},
                ],
            },
        ]

        pred2 = inference(conv2_with_history, model, tokenizer, processor, use_placeholder=False, topk=0, conv_iteration=2)
        generation_list.append(pred2["output_text"])
        item['generation_list'] = generation_list

        answer = extract_answer(pred2["output_text"])

        if args.dataset_name == "POPE":
            if parse_yesorno_ans(answer.lower()) == ground_truth.lower():
                score = 1
            else:
                score = 0
        elif args.dataset_name == "RealworldQA":
            if answer.lower() == ground_truth.lower() or answer.lower()[0] == ground_truth.lower()[0]:
                score = 1
            else:
                score = 0
        else:
            try:
                if answer.lower()[0] == ground_truth.lower()[0]:
                    score = 1
                else:
                    score = 0
            except:
                score = 0
        item["score"] = score
        output_data.append(item)
        if rank == 0:
            tbar.update(len(output_data) - done_count)
            done_count = len(output_data)
    return output_data

def main():
    
    args = parse_args()
    args.distributed = True
    args.dist_url = "env://"
    
    init_distributed_mode(args)
    rank, world_size = get_rank(), get_world_size()

    model_path = args.model_path
    output_file = args.output_file
    dataset_path = args.dataset_path
    local_result = run(model_path, dataset_path, rank, args, world_size)

    gather_list = [None] * world_size
    dist.all_gather_object(gather_list, local_result)  

    if rank == 0:
        all_items = []
        with open(output_file, "w", encoding="utf-8") as f:
            for res in gather_list:
                for item in res:  
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    all_items.append(item)

        dataset_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        for item in all_items:
            dataset = item.get("Category", "unknown")
            acc = item.get("score", 0)
            dataset_stats[dataset]["correct"] += acc
            dataset_stats[dataset]["total"] += 1

        print("\n=== Dataset Accuracy Summary ===")
        for name, stats in dataset_stats.items():
            avg_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{name}: Accuracy = {avg_acc:.4f}")

        total_correct = sum(stats["correct"] for stats in dataset_stats.values())
        total_count = sum(stats["total"] for stats in dataset_stats.values())
        overall_acc = total_correct / total_count if total_count > 0 else 0
        print(f"Overall Accuracy = {overall_acc:.4f}")
        
        print(f"{output_file}")
    
if __name__ == "__main__":
    main()