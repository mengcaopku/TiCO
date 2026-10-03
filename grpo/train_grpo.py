import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset

from trainer import Qwen2VLGRPOTrainer
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from datasets import Dataset, DatasetDict
from accuracy_reward_functions import v7w, docVQA, Math_eureka, ocr_and_freeform, openend_datasets, deepeyes_visual_toolbox, output_json_for_selection

dataset_handlers = {

    "v7w": (v7w, False),
    "gqa": (v7w, False),
    "openimages": (v7w, False),
    "CLEVR": (Math_eureka, False),
    "docvqa": (ocr_and_freeform, True),
    "infographicsvqa": (ocr_and_freeform, True),
    "textcap": (ocr_and_freeform, True), 
    "textvqa": (ocr_and_freeform, True), 
    "dude": (ocr_and_freeform, True),
    "sroie": (ocr_and_freeform, True), 
    "flickr30k": (ocr_and_freeform, True),
    "vsr": (openend_datasets, False),
    "cub": (openend_datasets, False),
    "visual_toolbox_v2": (deepeyes_visual_toolbox, False),
    "data_thinklite_reasoning_acc": (Math_eureka, False),
    "minio3": (deepeyes_visual_toolbox, False)
}

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    score_funcs: list[str] = field(
        default_factory=lambda: ["format"],
    )

    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    num_generations_stage1: Optional[int] = field(
        default=8,
        metadata={"help": "generations in stage1"},
    )

def accuracy_reward_stage2(completions, solution, dataset, image, problem_id, **kwargs):

    completion_contents_stage2 = [completion[-1] for completion in completions]

    rewards = []

    for content_stage2, sol in zip(completion_contents_stage2, solution):
        content_stage2 = content_stage2.split("assistant\n", 1)[-1].split("<|im_end|>", 1)[0]
        if isinstance(sol, list):
            pass
        else:
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

        if dataset[0] in dataset_handlers:
            handler_func, needs_dataset_arg = dataset_handlers[dataset[0]]
            if needs_dataset_arg:
                reward, student_answer = handler_func(content_stage2, ground_truth, dataset[0])
            else:
                reward, student_answer = handler_func(content_stage2, ground_truth)
        else:
            print(f"Error: no handler_func for dataset: {dataset[0]}")

        rewards.append(reward)
    return rewards

def format_reward_stage1_pointerpad(completions, **kwargs):

    completion_contents_stage1 = [completion[0] for completion in completions]
    def _compute_stage1_reward(content):
        return 1.0 if content=="<|Ground|>" else 0.0
    
    format_rewards = []
    for completion in completion_contents_stage1:
        format_reward = _compute_stage1_reward(completion)
        format_rewards.append(format_reward)
    return format_rewards


reward_funcs_registry = {
    "accuracy": accuracy_reward_stage2,
    "format": format_reward_stage1_pointerpad,
}

log_score_funcs_regitry = {
    "format": format_reward_stage1_pointerpad,
}



def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    score_funcs = [log_score_funcs_regitry[func] for func in script_args.score_funcs]

    if script_args.dataset_name[-6:] == '.jsonl':
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    STAGE_ONE_TEMPLATE = (
        "{Question} Your task is to locate the region that corresponds to the question. To indicate the region, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: <your_special_token_here>."
    )
    STAGE_TWO_TEMPLATE = (
        "{Question} Refer to the region extracted in the previous turn and provide the final answer."
    )

    def make_conversation_image(example):
        
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": STAGE_ONE_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }


    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(make_conversation_image)
        
    elif "video_filename" in dataset[script_args.dataset_train_split].features:
        print("has video in dataset")
        dataset = dataset.map(make_conversation_video)

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    trainer_cls = Qwen2VLGRPOTrainer
    print("using: ", trainer_cls)

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        score_funcs=score_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        num_generations_stage1=script_args.num_generations_stage1,
    )

    from torch.serialization import add_safe_globals
    import numpy as np
    np_core = np._core if hasattr(np, "_core") else np.core
    add_safe_globals([
        np_core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype(np.uint32)),
    ])

    last_checkpoint = None
    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_state(training_args.output_dir)
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)