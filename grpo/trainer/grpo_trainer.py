from PIL import Image,ImageDraw
import re

import os
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import itertools
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig

from trainer.image_pro import process_vision_info
import copy
import numpy as np
import gc
from tico.modeling_qwen25vl import Qwen2_5_VLForConditionalGenerationWithPointer
from tico.inference import inference_rl_batch
import matplotlib.pyplot as plt
from deepspeed import zero

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
ScoreFunc = Union[str, PreTrainedModel, Callable[[list, list, list, list, list], list[float]]]

STAGE_ONE_TEMPLATE = (
    "{Question} Your task is to locate the region that corresponds to the question. To indicate the region, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: <your_special_token_here>."
)
STAGE_TWO_TEMPLATE = (
    "{Question} Refer to the region extracted in the previous turn and provide the final answer."
)


def build_training_batch(pred_list, device):
    B = len(pred_list)
    ptr_action_patch = torch.zeros(B, dtype=torch.long, device=device)
    valid_pointer = torch.zeros(B, dtype=torch.bool, device=device)

    for i,p in enumerate(pred_list):
        if p.get("valid_pointer", False):
            valid_pointer[i] = True
            ptr_action_patch[i] = int(p["ptr_action_patch"])
        else:
            ptr_action_patch[i] = 0

    return {
        "ptr_action_patch": ptr_action_patch,
        "valid_pointer": valid_pointer,      
    }



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
        print(f"Max Ratio cannot be Larger than 200---{bbox}")
        return []

    return [x1, y1, x2, y2]

def resize_image(image, resize_to_pixels=401408):
    image_width, image_height = image.size
    if (resize_to_pixels is not None) and ((image_width * image_height) != resize_to_pixels):
        resize_ratio = (resize_to_pixels / (image_width * image_height)) ** 0.5
        image_width_resized, image_height_resized = int(image_width * resize_ratio), int(image_height * resize_ratio)
        image = image.resize((image_width_resized, image_height_resized))
    return image

class Qwen2VLGRPOTrainer(Trainer):

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        score_funcs: Union[ScoreFunc, list[ScoreFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        score_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        num_generations_stage1: Optional[int] = 8,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass 
            elif isinstance(torch_dtype, str): 
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )

            model_init_kwargs["use_cache"] = (
                True
            )
            model_init_kwargs["torch_dtype"] = torch.bfloat16
            model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        print(model)

        for param in model.visual.parameters():
            param.requires_grad = False

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        if is_deepspeed_zero3_enabled():
            self.ref_model = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id)
            pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            processing_class.image_processor.max_pixels = max_pixels
            processing_class.image_processor.min_pixels = min_pixels            

        chat_template_new = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}Picture {{ image_count.value }}: <|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}Video {{ video_count.value }}: <|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        processing_class.chat_template = chat_template_new

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        for i, score_func in enumerate(score_funcs):
            if isinstance(score_func, str):
                score_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    score_func, num_labels=1, **model_init_kwargs
                )
        self.score_funcs = score_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Score processing class
        if score_processing_classes is None:
            score_processing_classes = [None] * len(score_funcs)
        else:
            if len(score_processing_classes) != len(score_funcs):
                raise ValueError("The number of score processing classes must match the number of score functions.")
        self.score_processing_classes = score_processing_classes

        # Data collator
        def data_collator(features):
            return features

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length 
        self.num_generations = args.num_generations
        self.num_generations_stage1 = num_generations_stage1

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1.0, 
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,

            use_cache=True,
            repetition_penalty=1.05,
            bos_token_id=151643,
            eos_token_id=[151645, 151643],
        )
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True

        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):

        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    def _get_per_token_logps(self, model, input_ids, **kwargs):
        
        logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :] 
        input_ids = input_ids[:, 1:] 

        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def get_ptr_logp_new_batch_simple(
        self,
        model,
        tokenizer,
        input_ids,              # [B, L]
        ptr_action_patch,       # [B]
        valid_pointer=None,     # [B] bool or None
        patch_sample_tau=1.0,
        **stage1_inputs
    ):
        device = input_ids.device
        B, L = input_ids.shape

        if valid_pointer is None:
            valid_pointer = torch.ones(B, dtype=torch.bool, device=device)
        else:
            valid_pointer = valid_pointer.to(device)

        ptr_action_patch = ptr_action_patch.to(device)

        Ground_id = getattr(model.config, "Ground_token_id", None)
        if Ground_id is None:
            Ground_id = tokenizer.encode("<|Ground|>")[0]
        image_pad_id = tokenizer.encode("<|image_pad|>")[0]

        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            **stage1_inputs
        )
        hidden_last = outputs.hidden_states[-1]  # [B, L, H]
        H = hidden_last.shape[-1]

        head = model.multi_patch_pointer_head

        was_training = head.training
        head.eval()

        scaling = head.d_model ** 0.5

        dummy_h_ptr = hidden_last[0, 0].view(1, H)   # [1, H]
        dummy_img   = hidden_last[0, 0].view(1, H)   # [1, H]

        logp_list = []

        for i in range(B):
            ids = input_ids[i]        # [L]
            hs  = hidden_last[i]      # [L, H]

            ptr_pos = torch.nonzero(ids == Ground_id, as_tuple=False).squeeze(-1)
            if ptr_pos.numel() > 0:
                ptr_pos = int(ptr_pos[0].item())
                h_ptr = hs[ptr_pos].view(1, H)      # [1, H]
            else:
                h_ptr = dummy_h_ptr
                valid_pointer[i] = False

            img_mask = (ids == image_pad_id)
            if bool(img_mask.any().item()):
                image_embeds = hs[img_mask]         # [n_enc, H]
            else:
                image_embeds = dummy_img            # [1, H]
                valid_pointer[i] = False

            proj_enc = head.projection_enc(image_embeds)         # [n_enc, H]
            proj_dec = head.projection_dec(h_ptr)                # [1, H]
            patch_logits = (proj_dec @ proj_enc.transpose(0, 1)) / scaling  # [1, n_enc]
            attn_scores = torch.softmax(patch_logits, dim=-1)[0]            # [n_enc]

            a = int(ptr_action_patch[i].item())

            if bool(valid_pointer[i].item()):
                p_patch = attn_scores.clamp(min=0.0)
                p_patch = p_patch / p_patch.sum().clamp_min(1e-12)

                if patch_sample_tau is not None and abs(float(patch_sample_tau) - 1.0) > 1e-6:
                    logits = torch.log(p_patch.clamp_min(1e-12))
                    p_patch = torch.softmax(logits / float(patch_sample_tau), dim=-1)

                if 0 <= a < p_patch.numel():
                    logp_i = torch.log(p_patch[a].clamp_min(1e-12))
                else:
                    valid_pointer[i] = False
                    print("ptr_action_patch out of p_patch range!")
                    logp_i = patch_logits.sum() * 0.0
            else:
                logp_i = patch_logits.sum() * 0.0

            logp_list.append(logp_i)

        ptr_logp_new = torch.stack(logp_list, dim=0)

        head.train(was_training)

        return ptr_logp_new, valid_pointer

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def _prepare_for_stage2(self, origin_prompt, origin_problem, combined_images, image_stage2, bbox):

        next_stage_entry = {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": STAGE_TWO_TEMPLATE.format(Question=origin_problem)
                },
            ],
        }

        origin_prompt.extend([
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(bbox).replace("<|im_end|>", "")}]
        },
        copy.deepcopy(next_stage_entry)
        ])

        combined_images.append(image_stage2)
        return origin_prompt, combined_images

    def _save_pred_products(
        self,
        base_name: str,
        image: Image.Image,
        pred: dict,
        output_dir: str,
        prefix: str = "",
        save_image: bool = True,
    ):

        w, h = image.size

        default_bbox = (0.4, 0.4, 0.5, 0.5)

        valid_pointer = pred.get("valid_pointer", True)

        chosen_bbox = pred.get("chosen_bbox", None)
        pred_boxes = pred.get("bboxes", None)
        if chosen_bbox is not None:
            bbox0 = chosen_bbox
        elif isinstance(pred_boxes, (list, tuple)) and len(pred_boxes) > 0:
            bbox0 = pred_boxes[0]
        else:
            bbox0 = default_bbox

        if not valid_pointer:
            pass

        try:
            img_with_bbox = draw_pred_bbox(image, w, h, pred_boxes)
        except Exception:
            img_with_bbox = image 

        x0, y0, x1, y1 = bbox0
        left, top = int(x0 * w), int(y0 * h)
        right, bottom = int(x1 * w), int(y1 * h)

        left, top = max(0, left), max(0, top)
        right, bottom = min(w, right), min(h, bottom)

        min_size = 28
        crop_w = right - left
        crop_h = bottom - top

        if crop_w < min_size or crop_h < min_size:
            cx = (left + right) // 2
            cy = (top + bottom) // 2

            half_w = max(crop_w // 2, min_size // 2)
            half_h = max(crop_h // 2, min_size // 2)

            left = cx - half_w
            right = cx + half_w
            top = cy - half_h
            bottom = cy + half_h

            left = max(0, left)
            top = max(0, top)
            right = min(w, right)
            bottom = min(h, bottom)

            if right - left < min_size:
                if left == 0:
                    right = min(w, left + min_size)
                else:
                    left = max(0, right - min_size)

            if bottom - top < min_size:
                if top == 0:
                    bottom = min(h, top + min_size)
                else:
                    top = max(0, bottom - min_size)

        img_crop_bbox = image.crop((left, top, right, bottom))

        n_width, n_height = pred.get("n_width", None), pred.get("n_height", None)
        attn_scores = pred.get("attn_scores", None)
        att_map = None
        if attn_scores is not None and n_width is not None and n_height is not None:
            try:
                att_map = get_attn_map(image, attn_scores, n_width, n_height, item=None)
            except Exception:
                att_map = None

        bbox_image_path = os.path.join(output_dir, f"{base_name}_{prefix}with_bbox.png")
        attn_image_path = os.path.join(output_dir, f"{base_name}_{prefix}attention.png")
        crop_image_path = os.path.join(output_dir, f"{base_name}_{prefix}_crop.png")
        chosen_box_idx = pred.get("chosen_bbox_idx", 0)
        if chosen_box_idx != 0:
            print(chosen_box_idx)

        if save_image:
            os.makedirs(output_dir, exist_ok=True)
            try:
                img_with_bbox.save(bbox_image_path)
            except Exception as e:
                pass
            if att_map is not None:
                try:
                    att_map.save(attn_image_path)
                except Exception:
                    pass
            try:
                img_crop_bbox.save(crop_image_path)
            except Exception:
                pass

        return img_crop_bbox

    def _generate_for_stage2_batch(self, messages_stage2_list, all_images_stage2_list):
        
        texts = [
            self.processing_class.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, add_vision_id=True)
            for msg in messages_stage2_list
        ]
        prompt_stage2_inputs = self.processing_class(
            text=texts,
            images=all_images_stage2_list,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_stage2_inputs = super()._prepare_inputs(prompt_stage2_inputs)
        return prompt_stage2_inputs        

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        origin_problem = inputs[0]["problem"]
        prompts = [x["prompt"] for x in inputs]
        prompts_text = self.processing_class.apply_chat_template(inputs[0]['prompt'],tokenize=False, add_generation_prompt=True, add_vision_id=True)

        if "image" in inputs[0]:
            images = []
            for (cur_idx, cur_input) in enumerate(inputs):
                copy_input = cur_input.copy()
                copy_input['prompt'][0]['content'][0]['image'] = inputs[cur_idx]["image"]
                images.append(process_vision_info(copy_input["prompt"])[0])

            images = images[0]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images if "image" in inputs[0] else None,
            videos=None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            num_generations = self.generation_config.num_return_sequences
            temp_generation_config = copy.deepcopy(self.generation_config)
            temp_generation_config.num_return_sequences = self.num_generations_stage1

            all_completions_stage1 = [None] * num_generations
            all_completions = [None] * num_generations

            prompt_for_generation = [inputs[0]['prompt'][:] for _ in range(num_generations)]

            all_images_final_list = [None] * num_generations
            for i in range(num_generations):

                all_images_final_list[i] = copy.deepcopy(images)

            with torch.no_grad():

                # Stage 1 inference
                pred_list_stage1 = inference_rl_batch(
                    prompt_inputs, temp_generation_config, unwrapped_model,
                    self.processing_class.tokenizer, use_placeholder=False, topk=5, conv_iteration=1
                )

                original_list = [p.get("output_text", "") for p in pred_list_stage1]

                output_text_stage1_list = list(itertools.islice(itertools.cycle(original_list), num_generations))

                gc.collect()
                torch.cuda.empty_cache()

                messages_stage2_list = []

                for idx in range(num_generations):

                    p_idx = pred_list_stage1[idx % len(pred_list_stage1)]

                    all_completions_stage1[idx] = p_idx.get("generated_ids_list").unsqueeze(0)

                    img_with_bboxes_stage2 = self._save_pred_products(
                        base_name=f"{origin_problem[0:20]}_{idx}",
                        image=images[0],
                        pred=p_idx,
                        output_dir="",
                        prefix=idx,
                        save_image=False
                    )
                    w_crop, h_crop = img_with_bboxes_stage2.size

                    if w_crop * h_crop > self.processing_class.image_processor.max_pixels:
                        img_with_bboxes_stage2 = resize_image(img_with_bboxes_stage2)

                    prompt_for_generation[idx], all_images_final_list[idx] = self._prepare_for_stage2(
                        prompt_for_generation[idx],
                        origin_problem,
                        all_images_final_list[idx],
                        img_with_bboxes_stage2,
                        output_text_stage1_list[idx]
                    )
                    messages_stage2_list.append(prompt_for_generation[idx])

                if messages_stage2_list:
                    prompt_stage2_inputs = self._generate_for_stage2_batch(prompt_for_generation, all_images_final_list)
                    temp_stage2_config = copy.copy(temp_generation_config)
                    temp_stage2_config.num_return_sequences = 1
                else:
                    prompt_stage2_inputs = None
                    temp_stage2_config = None

                # ---- Stage2 inference ----
                if prompt_stage2_inputs:

                    pred_list_stage2 = inference_rl_batch(
                        prompt_stage2_inputs, temp_stage2_config, unwrapped_model,
                        self.processing_class.tokenizer, use_placeholder=False, topk=5, conv_iteration=2
                    )

                    for idx in range(len(pred_list_stage2)):
                        gen = pred_list_stage2[idx].get("generated_ids_list")
                        all_completions[idx] = gen.unsqueeze(0)

                    del pred_list_stage2
                else:
                    pass

                del output_text_stage1_list, prompt_for_generation

                gc.collect()
                torch.cuda.empty_cache()

            max_length = max(completion.size(1) for completion in all_completions)
            padded_completions = []
            indices_list = []

            for completion in all_completions:

                _seq = completion[0]
                start_pattern = torch.tensor([151644, 77091, 198], device=_seq.device)
                end_token = torch.tensor(151645, device=_seq.device)

                sample_indices = []
                start_indices = (_seq.unfold(0, len(start_pattern), 1) == start_pattern).all(dim=1).nonzero(as_tuple=True)[0]
                for start_index in start_indices:
                    start_index = start_index.item() + len(start_pattern)
                    end_indices = (_seq[start_index:] == end_token).nonzero(as_tuple=True)[0]
                    end_index = start_index + (end_indices[0].item() + 1 if end_indices.numel() > 0 else len(_seq) - start_index)
                    sample_indices.append(list(range(start_index, end_index)))
                indices_list.append(sample_indices)

                if completion.size(1) < max_length:
                    padding = torch.full(
                        (completion.size(0), max_length - completion.size(1)),
                        self.processing_class.tokenizer.pad_token_id,
                        dtype=completion.dtype,
                        device=completion.device,
                    )
                    padded_completion = torch.cat([completion, padding], dim=1)
                else:
                    padded_completion = completion
                padded_completions.append(padded_completion)

            max_length_stage1 = max(completion_stage1.size(1) for completion_stage1 in all_completions_stage1)
            padded_completions_stage1 = []
            for completion_stage1 in all_completions_stage1:
                if completion_stage1.size(1) < max_length_stage1:
                    padding_stage1 = torch.full(
                        (completion_stage1.size(0), max_length_stage1 - completion_stage1.size(1)),
                        self.processing_class.tokenizer.pad_token_id,
                        dtype=completion_stage1.dtype,
                        device=completion_stage1.device,
                    )
                    padded_completion_stage1 = torch.cat([completion_stage1, padding_stage1], dim=1)
                else:
                    padded_completion_stage1 = completion_stage1
                padded_completions_stage1.append(padded_completion_stage1)
            prompt_completion_ids_stage1 = torch.cat(padded_completions_stage1, dim=0)

            prompt_completion_ids = torch.cat(padded_completions, dim=0)

        completion_ids = prompt_completion_ids[:]
        
        device = self.accelerator.device
        batch_size, seq_length = completion_ids.size()
        completion_mask = torch.zeros((batch_size, seq_length), dtype=torch.int, device=device)

        for i, indices in enumerate(indices_list):
            for index_range in indices:
                completion_mask[i, index_range] = 1 

        prompt_stage2_inputs.pop("input_ids")
        prompt_stage2_inputs.pop("attention_mask")
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")

        batch = build_training_batch(pred_list=pred_list_stage1, device=prompt_completion_ids.device)
        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_stage2_inputs)

        prompt_inputs_rep = {
            k: (v.repeat(num_generations, *([1] * (v.dim() - 1))) if isinstance(v, torch.Tensor) else v)
            for k, v in prompt_inputs.items()
        }
        ptr_logp_new, batch["valid_pointer"] = self.get_ptr_logp_new_batch_simple(
            model=model,
            tokenizer=self.processing_class.tokenizer,
            input_ids=prompt_completion_ids_stage1,    
            ptr_action_patch=batch["ptr_action_patch"],    
            valid_pointer=batch.get("valid_pointer", None),
            patch_sample_tau=batch.get("patch_sample_tau", 1.0),
            **prompt_inputs_rep                       
        )

        completion_mask = completion_mask[:, 1:]

        with torch.inference_mode():
            if self.ref_model is not None:

                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_stage2_inputs)

            else:
                with self.accelerator.unwrap_model(model).disable_adapter():

                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_stage2_inputs)

        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False)

        if is_conversational(inputs[0]):
            completions = [re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", completion, re.S) for completion in completions]


        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:

                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        scores_per_func = torch.zeros(len(prompts), len(self.score_funcs), device=device)
        for i, (score_func, score_processing_class) in enumerate(
            zip(self.score_funcs, self.score_processing_classes)
        ):
            score_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in score_kwargs:
                for example in inputs:
                    score_kwargs[key].extend([example[key]] * self.num_generations)
            
            output_score_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            scores_per_func[:, i] = torch.tensor(output_score_func, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        token_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        if "valid_pointer" in batch: 
            valid = batch["valid_pointer"].to(ptr_logp_new.device).float()
        else: 
            print("No Valid Pointer in Batch!")
            valid = torch.ones_like(ptr_logp_new, dtype=torch.float)

        vsum = valid.sum()
        if vsum.item() == 0:
            ptr_loss = (ptr_logp_new.sum() * 0.0)
        else:
            sur = torch.exp(ptr_logp_new - ptr_logp_new.detach()) * advantages
            sur = sur * valid
            ptr_loss = -(sur.sum() / vsum)

        lambda_ptr = getattr(self, "lambda_ptr", 1.0)
        loss = token_loss + lambda_ptr * ptr_loss

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        score_per_func = self.accelerator.gather_for_metrics(scores_per_func).mean(0)
        for i, score_func in enumerate(self.score_funcs):
            score_func_name = score_func.__name__
            self._metrics[f"scores/{score_func_name}"].append(score_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()