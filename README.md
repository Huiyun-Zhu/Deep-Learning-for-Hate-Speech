# EE559 Group Project – Group 47

## Abstract
Detection of hate-speech plays a crucial role in improving digital environment, where complicated and diverse emoticons are often used as the cover to evil intentions. This study delves into the interaction between textual content and emojis to understand how emojis influence the interpretation of language in online discussions. By analyzing extensive datasets of text and emojis, augmented for diversity, we aim to uncover patterns in how emojis alter the connotations and contexts of written communication. We then refine algorithms based on **DeBERTa** for detecting and categorizing offensive content. Ultimately, our research enhances the accuracy of detecting hate-speech on Chinese social media platforms and aspires to cultivate a digital environment characterized by inclusivity, tolerance, and mutual respect, fostering healthier online communities.  

**Keywords:**  
Hate Speech  
Emoji Semantics  
DeBERTa Model

---

## 📁 Project Structure
```

├── EE559-Group47/All the Data 
├── Code/
│   ├── Build_ch.py
│   ├── Weibo_emoji.ipynb
│   ├── Weibo_emoji.py
│   ├── Weibo_emoji_check.py
│   ├── Weibo_text.ipynb
│   └── Weibo_text.py
└── Result/
│    ├── result_build.csv
│    ├── result_build.png
│    ├── result_compare.csv
│    └── result_compare.png
│ 
├── Report
│ 
├── README.md
│ 
└── Slides

```

---

## 🚀 Process Overview

1. **Data Preprocessing & Model Evaluation**
   - `Weibo_text.ipynb` (or `Weibo_text.py` for GPU use): Evaluates DeBERTa model performance on Weibo text data.
   - `Weibo_emoji.ipynb` (or `Weibo_emoji.py` for GPU use): Evaluates DeBERTa model performance on Weibo emoji data.

2. **Model Training & Fine-Tuning**
   - `Build_ch.py`: Trains DeBERTa on the Build_emoji_ch dataset with 4 rounds of training to improve emoji data performance.

3. **Model Validation**
   - `Weibo_emoji_check.py`: Tests the trained model from step 2 on Weibo emoji data to verify performance improvement.

4. **Results**
   - `result_build.png` & `result_build.csv`: Shows training results from the 4 rounds of training in step 2.
   - `result_compare.png` & `result_compare.csv`: Compares model performance on:
     - Weibo text data (baseline)
     - Weibo emoji data (baseline)
     - Weibo emoji data after improvement through 4 training rounds.

---

## 📊 Data Access

The project dataset can be accessed via the following link:  
👉 [Google Drive Folder](https://drive.google.com/drive/folders/1o_lfWTs4UHc1FmPSv-JyWe2leNqUXQlS?usp=sharing)

---
