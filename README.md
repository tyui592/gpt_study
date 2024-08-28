GPT study
==

This is a repository for study GPT.
I implement a small size GPT model with a small text dataset.
- Number of parameters: 19,672,880.
  - $d_{model}$: 256
  - $d_{ff}$: 512
  - Number of heads: 16
  - Number of layers: 8
  - More details on default arguments in [config.py](./config.py)
- Dataset: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
  - I made a token id array for efficient training and validation.
  - Please see the [notebook](./DataPreprocessing.ipynb)

# Usage
Requirements
- ftfy version: 6.2.3
- spacy version: 3.7.5
- torchtext version: 0.18.0
- torch version: 2.3.0
- datasets version: 2.20.0
- wandb (Optional)

Example train script.
```bash
python main.py --save_root "./model-store/ex01" --print_interval 0.2 \
    --n_layers 8 --epoch 512 --context_size 128 \
    --wb_flag --wb_project 'gpt' --wb_notes "base model" --wb_tag "base"
```

Training Losses

<img width="893" alt="image" src="https://github.com/user-attachments/assets/087a4ea9-2ec9-4d5f-9d47-0c1c45397ee0">

Results (please check the [notebook](./TextGeneration.ipynb)
* Input: lily went to the store and bought a new toy.
* Output: lily went to the store and bought a new toy . she was so happy and could n't wait to play with it . but when she got home , she saw that her toy was missing . she looked everywhere but could n't find it . lily was sad and did n't know what to do . she asked her mom for help , but her mom did n't know how to fix it . lily was very sad and did n't know what to do . she went to her room and cried . her mom came in and asked what was wrong . lily told her about her missing toy . her mom said , " do n't worry , we can fix it together . " they worked together and lily was able to find her toy . she was so happy again . she hugged her mom was so happy again . lily was so happy again . she was so glad that she was n't have to be sad . from now . she was n't have to lose her toy . she was n't have to lose it . lily . she was so sad . she was so sad . her mom was so sad . her mom was so sad . her mom was so sad . her mom was so sad . her mom was so sad . she was so sad . her mom was so sad . her mom was so sad . she
