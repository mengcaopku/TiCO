# Thinking with Images without Coordinates

PyTorch implementation of paper:

> **Thinking with Images without Coordinates**
>
> Meng Cao\*, Haoze Zhao\*, Yi Wang, Ian Reid and Xiaodan Liang.
>
> [Project Page]() [ArXiv]()

This package contains the training and evaluation code for **TiCO: Thinking with Images without Coordinates**, together with the processed instruction files used by the SFT and GRPO stages.

## Repository Structure

```text
TiCO_Code/
├── data/
│   ├── prepare_data.py
│   ├── sft_data_config.yaml
│   ├── tico_sft.json
│   └── tico_grpo.jsonl
├── sft/tico/                 # SFT model, dataset, trainer, and inference code
├── grpo/                     # GRPO training code and reward functions
├── eval/
│   ├── dataset/              # Evaluation jsonl files
│   ├── output/               # Provided evaluation logs/results
│   ├── scripts/
│   └── eval_TiCO.sh
├── scripts/
│   ├── train_sft.sh
│   ├── train_grpo.sh
│   └── zero3.json
└── pyproject.toml
```

## Installation

```bash
cd TiCO_Code
conda create -n tico python=3.10
conda activate tico

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -e .
```

The default scripts use Qwen2.5-VL-7B-Instruct and DeepSpeed ZeRO-3. Please update local model and data paths before running the scripts.

## Data Preparation

### SFT Data

The processed SFT annotation file is provided at:

```text
data/tico_sft.json
```

Before SFT training, update:

```text
data/sft_data_config.yaml
scripts/train_sft.sh
```

Set the image root in `--image_folder` to the local directory containing images downloaded and processed from the `Mini-o3/Mini-o3-Coldstart-Dataset` Hugging Face dataset.

### GRPO Data

The processed GRPO annotation file is provided at:

```text
data/tico_grpo.jsonl
```

Before GRPO training, update the `--prefix` argument in:

```text
scripts/train_grpo.sh
```

The prefix should point to the local image directory downloaded and processed from the `ChenShawn/DeepEyes-Datasets-47k` Hugging Face dataset.

## SFT Training

Edit `scripts/train_sft.sh` and set:

```bash
llm_model="path/to/Qwen2.5-VL-7B-Instruct"
output_dir="./checkpoints/TiCO_SFT"
--image_folder "path/to/sft/images"
```

Then run:

```bash
bash scripts/train_sft.sh
```

## GRPO Training

The GRPO stage starts from the SFT checkpoint. Edit `scripts/train_grpo.sh` and set:

```bash
python data/prepare_data.py --prefix path/to/DeepEyes_Dataset/
--model_name_or_path "./checkpoints/TiCO_SFT"
--output_dir "./checkpoints/TiCO/"
```

Then run:

```bash
bash scripts/train_grpo.sh
```

## Inference and Evaluation

The evaluation script covers V* Bench and HR-Bench 4K/8K. Download and process the benchmark images from the following Hugging Face repositories:

- `DreamMr/HR-Bench`
- `lmms-lab/vstar-bench`

Then edit `eval/eval_TiCO.sh` and set:

```bash
MODEL_PATH="path/to/TiCO/checkpoint"
BENCH_ROOT="path/to/Eval_Dataset"
NPROC=8
```

Run evaluation with:

```bash
bash eval/eval_TiCO.sh
```

The script writes predictions to:

```text
eval/output/{HR_Bench_4K,HR_Bench_8K,vstar_bench}/TiCO.jsonl
```

Provided evaluation logs/results for V* Bench and HR-Bench 4K/8K are included under:

```text
eval/output/
```

## Notes

- The scripts assume a multi-GPU environment. Adjust `--nproc_per_node`, batch size, and DeepSpeed settings according to the available hardware.
- Large image files and base model checkpoints are not included in this supplement and should be downloaded separately.
- Paths in the scripts are placeholders and must be changed to match the local environment.
