# RL-VLM Evaluation Report

## Summary of Accuracy
| Model | Accuracy | Correct / Total | Avg Caption Length |
|-------|----------|-----------------|--------------------|
| PPO | 59.00% | 59 / 100 | 16.2 words |
| GRPO | 63.00% | 63 / 100 | 60.0 words |
| REINFORCE | 30.00% | 30 / 100 | 1.0 words |

## Qualitative Analysis (First 5 Samples)
### Sample 0
**Image:** ![Image](eval_images/000000012209.jpg)

**Question:** What is the player doing in the image?
   - A) He is catching a ball
   - B) He is running
   - C) He is swinging a bat
   - D) He is pitching

**Correct Answer:** C

**PPO** (Pred: C):
> A baseball player holding a bat on a field.

**GRPO** (Pred: C):
> A baseball player in a striped uniform is holding a bat and preparing to swing. The scene is set on a baseball field with a netted fence in the background. Spectators are visible in the foreground, some taking photos. The player is wearing a helmet and a belt. The field has a green grassy area and a dirt infield.

**REINFORCE** (Pred: B):
> ,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy

---
### Sample 1
**Image:** ![Image](eval_images/000000012209.jpg)

**Question:** What is the player's uniform number?
   - A) 13
   - B) 15
   - C) 14
   - D) 12

**Correct Answer:** C

**PPO** (Pred: D):
> A baseball player holding a bat on a field.

**GRPO** (Pred: D):
> A baseball player in a striped uniform is holding a bat and preparing to swing. The scene is set on a baseball field with a netted fence in the background. Spectators are visible in the foreground, some taking photos. The player is wearing a helmet and a belt. The field has a green grassy area and a dirt infield.

**REINFORCE** (Pred: D):
> ,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy

---
### Sample 2
**Image:** ![Image](eval_images/000000012209.jpg)

**Question:** What is the position of the player in the image?
   - A) Pitcher
   - B) Catcher
   - C) Base runner
   - D) Baseball player

**Correct Answer:** D

**PPO** (Pred: D):
> A baseball player holding a bat on a field.

**GRPO** (Pred: D):
> A baseball player in a striped uniform is holding a bat and preparing to swing. The scene is set on a baseball field with a netted fence in the background. Spectators are visible in the foreground, some taking photos. The player is wearing a helmet and a belt. The field has a green grassy area and a dirt infield.

**REINFORCE** (Pred: D):
> ,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy

---
### Sample 3
**Image:** ![Image](eval_images/000000012209.jpg)

**Question:** What is the atmosphere of the scene in the background?
   - A) Calm and peaceful
   - B) Excited and energetic
   - C) Dull and uninteresting
   - D) Dark and ominous

**Correct Answer:** B

**PPO** (Pred: B):
> A baseball player holding a bat on a field.

**GRPO** (Pred: B):
> A baseball player in a striped uniform is holding a bat and preparing to swing. The scene is set on a baseball field with a netted fence in the background. Spectators are visible in the foreground, some taking photos. The player is wearing a helmet and a belt. The field has a green grassy area and a dirt infield.

**REINFORCE** ( Pred: C):
> ,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy

---
### Sample 4
**Image:** ![Image](eval_images/000000022411.jpg)

**Question:** What is the number of students in the image?
   - A) 5
   - B) 6
   - C) 7
   - D) 8

**Correct Answer:** B

**PPO** ( Pred: C):
> A group of people sitting in a library, eating pizza and talking.

**GRPO** (Pred: B):
> A group of teenagers is gathered in a library, sitting at tables and eating pizza. The room is well-lit with natural light coming in through large windows, and there are bookshelves filled with books in the background. The atmosphere is casual and relaxed, with some students standing and others sitting. The teenagers are engaged in conversation and enjoying their meal together.

**REINFORCE** ( Pred: C):
> ,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy,copy

---
