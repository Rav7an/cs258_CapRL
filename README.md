# CapRL: Reinforcement Learning for Vision-Language Models

This project implements and evaluates various Reinforcement Learning (RL) algorithms for fine-tuning Vision-Language Models (VLMs), specifically focusing on Qwen2-VL.

## Project Structure

- **`src/`**: Contains the core source code for training and evaluation.
  - `train_ppo.py`, `train_ppo2.py`, `train_ppo3.py`: Scripts for training with Proximal Policy Optimization (PPO).
  - `train_grpo.py`: Script for training with Group Relative Policy Optimization (GRPO).
  - `train_reinforce.py`: Script for training with REINFORCE.
  - `evaluate_all_models.py`: Script to evaluate trained models.
  - `baseline.py`: Baseline model implementation.
- **`dataset_generation/`**: Contains notebooks (e.g., `GenQnA.ipynb`) for generating the Q&A dataset.
- **`output/`**: Stores training logs, model checkpoints, and evaluation results.
- **`plots/`**: Directory for generated plots and visualizations. (depricated file can be ignored)

## Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train a model using PPO:
```bash
python src/train_ppo.py
```

To train a model using GRPO:
```bash
python src/train_grpo.py
```

To train a model using REINFORCE:
```bash
python src/train_reinforce.py
```

### Evaluation

To evaluate the trained models:
```bash
python src/evaluate_all_models.py
```

## Dataset
This is the main source of our dataset generation -> (../CapRL/sharegpt_training_5k.json)
Using the menitoned file we have downloaded the images from coco and generated the mcq dataset mentionde below using the dataset_generation/GenQnA.ipynb. We have not included the images data in here. 

The project uses a custom dataset located at `caprl_mcq_dataset_final.jsonl`.

## Finetuned Models.
R mail access is given to the below mentioned link for finetuned models.

https://drive.google.com/drive/folders/1ViEzBXDk5ADt-cJkjWN_2I4hmVXcsuHc?usp=sharing