# EE559 Group Mini Project, Group 47

Xuhang Liu

Pu Zhi

Huiyun Zhu

## Project Sturcture

├── README.md
	├── code
	│   ├── Build_ch.py
	│   ├── Weibo_emoji.ipynb
	│   ├── Weibo_emoji.py
	│   ├── Weibo_emoji_check.py
	│   ├── Weibo_text.ipynb
	│   └── Weibo_text.py
	├── result
	│   ├── result_build.csv
	│   ├── result_build.png
	│   ├── result_compare.csv
	│   └── result_compare.png

## Process

1. Weibo_text.ipynb (Weibo_text.py on GPU) - Check the performance of DeBERTa on weibo_text data.
2. Weibo_emoji.ipynb (Weibo_emoji.py on GPU) - Check the performance of DeBERTa on weibo_emoji data.
3. Build_ch.py - Training DeBERTa model on 4 rounds with Build_emoji_ch data to improve the performance of DeBERTa on emoji data.
4. Weibo_emoji_check.py - Use the trained model from 3 to check whether the model is improved on the weibo_emoji data.
5. result_build.png/result_build.csv - The training result on 4 rounds in 3.
6. result_compare.png/result_compare.csv - The results from model evaluate on weibo_text, weibo_emoji previously and weibo_emoji after improving through 4 rounds.

## Data

The data could be accessed through:

https://drive.google.com/drive/folders/1o_lfWTs4UHc1FmPSv-JyWe2leNqUXQlS?usp=sharing
