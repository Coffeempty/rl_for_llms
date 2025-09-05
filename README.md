# RL for LLMs – Reinforcement Learning Fine-Tuning with GRPO + RULER

[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-🔥-ee4c2c)](https://pytorch.org/)  
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/)  
[![OpenPipe ART](https://img.shields.io/badge/OpenPipe-ART-blue)](https://openpipe.ai/blog/ruler)  
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)  

---

## 🚀 Overview

This repository explores **reinforcement learning fine-tuning (RLFT)** for large language models using the **GRPO (Generalized REINFORCE with Policy Optimization)** algorithm.  

Unlike traditional RLHF pipelines that rely on costly labeled datasets, this project uses a **synthetic data pipeline** powered by:  
- **OpenRouter API** → Generates synthetic training/test prompts from LLMs.  
- **RULER (rule-based reward evaluation)** → Scores outputs against task-specific criteria.  
- **GRPO** → Optimizes the model using stable, KL-regularized policy gradients.  

✨ The result is a **closed-loop system** where the model can train itself on **any task** simply by editing the **task description** in the notebook.  

---

## 🔄 How It Works

1. **Task Definition**  
   - Specify your desired behavior in the notebook (e.g., *“Rewrite text in a given style”*).  

2. **Synthetic Data Generation**  
   - Prompts are dynamically generated via **OpenRouter API**.  
   - The model **generates its own training and test samples**.  

3. **RULER Evaluation**  
   - Outputs are evaluated against a **rule-based scoring system**.  
   - Rules can enforce **style, tone, length, coherence, or structure**.  
   - Evaluation can be **deterministic** (programmatic rules) or **LLM-assisted**.  

4. **GRPO Training**  
   - GRPO applies policy gradients with KL regularization for stability.  
   - Rewards from RULER guide the optimization loop.  

📊 *[Insert Pipeline Diagram: Prompt → Model → RULER → GRPO Update → Fine-tuned Model]*  

---

## ✨ Key Features

- **Synthetic data pipeline** – no datasets required.  
- **Task-agnostic training** – modify the task description to adapt training.  
- **Modular design** – swap prompt-generation and evaluation models as needed.  
- **Stable optimization** – GRPO prevents catastrophic drift.  
- **Style-controlled generation** – output text in different tones while preserving meaning.  

---

## 📚 Background

### 🔹 GRPO in RL Fine-Tuning
GRPO is an RL algorithm that extends REINFORCE with:  
- **KL divergence penalty** to constrain the policy near the base model.  
- **Reward shaping** to capture nuanced preferences.  
- **Variance reduction** via baselines.  

This makes GRPO a **lighter and more stable alternative to PPO** for LLM fine-tuning.  

### 🔹 RULER (Rule-based Rewards)
[RULER (OpenPipe)](https://openpipe.ai/blog/ruler) is a **rule-based evaluation framework** for LLM training.  

- Encodes **programmatic reward functions** (e.g., adherence to style, structure).  
- Allows **fine-grained, interpretable control**.  
- Can use an **LLM as a grader** for more subjective tasks.  

This repo leverages RULER as the **reward function backbone** for RL fine-tuning.  

### 🔹 Related Work
- **ART: RL for LLMs** – modular RLHF training toolkit.  
- **RULER (OpenPipe)** – rule-based evaluation framework.  
- **PPO vs GRPO** – PPO dominates RLHF, but GRPO offers simpler, compute-friendly optimization.  

---

## 📊 Training & Results

This project currently includes **one training run**.  

### Metrics Tracked
- **Reward progression**  
- **Policy loss**  
- **KL divergence**  

📈 *[Insert Graph: Reward vs Step]*  
📉 *[Insert Graph: KL Divergence vs Step]*  
📊 *[Insert Graph: Loss vs Step]*  
🪙 *[Insert Graph: Policy Loss vs Step]*  

Logs and visualizations are stored in the `assets/` folder.  

---

## 🔬 Inference Showcase

After training, the model was tested on **unseen prompts** with style-transfer objectives.  

### Example Runs

| **Test** | **Input** | **Requested Style** | **Model Output** |
|----------|-----------|----------------------|------------------|
| 1 | The server is down, which means nobody can log in to work remotely. | Humorous Gen-Z slang | The server is down, so we're all stuck at home, unable to log in remotely. Please fix the situation pronto. |
| 2 | I regret to inform you that your manuscript does not meet our publication standards. | Shakespearean tragic monologue | Ah, poor soul, alas, I must inform you that your work does not please our esteemed peers. We, the humble editors, are but a few, yet we do our best to discern the finest sentiments amidst the myriad of submissions. Please accept this as a sign of our respect and admiration for your literary endeavors. It is with a heavy heart that we must part ways, for we do not believe your manuscript to be worthy of our esteemed journal's prestigious publication.|
| 3 | Hey dude, wanna grab burritos later? Maybe around 7? | Diplomatic correspondence |  Dear [Recipient's Name], Thank you for reaching out. I would be delighted to meet for burritos at approximately 7 PM. Please let me know if you are available at that time, and we can discuss further details. Best regards, [Your Name]|
| 4 | The cat knocked over my coffee mug and now there's liquid all over my keyboard. | Haiku | The cat again spills / my coffee, now the keyboard's / covered in brown ink. |
| 5 | We are writing to confirm receipt of your application. | Sarcastic breakup text | Congratulations, you’ve officially been dumped. |

---

### 🎉 Capability Summary

The fine-tuned model **`rl-model-001`** can:  
- Dynamically generate its own synthetic data.  
- Rewrite text in requested styles while preserving meaning.  
- Adapt to **any task** with a simple task description change.  

**Example:**  
- Input: *“I had a good day.”*  
  - Sarcastic → *“Oh yeah, because absolutely nothing screams excitement like eating instant noodles alone all day.”*  
  - Poetic → *“The sun caressed my weary soul, and joy bloomed quietly within.”*  

---

## 🛠️ Getting Started

### Requirements
This repo uses:  
- [`weave==0.50.0`](https://pypi.org/project/weave/)  
- [`openpipe-art[backend]==0.4.11`](https://pypi.org/project/openpipe-art/)  
- `torch`, `transformers`, `datasets`, etc.  

### Setup
```bash
git clone https://github.com/Coffeempty/rl_for_llms.git
cd rl_for_llms
