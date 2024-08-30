GPT study
==

This is a repository to study GPT.  I implement a small size GPT model with a smaller text dataset.
- Number of parameters: 19,672,880.
  - $d_{model}$: 256
  - $d_{ff}$: 512
  - Number of heads: 16
  - Number of layers: 8
  - More details on default arguments in [config.py](./config.py)
- Dataset: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
  - I made a token id array for efficient training and validation.
  - Please see the [notebook](./DataPreprocessing.ipynb)
  - NOTE: Since TinyStories has a lot of training data and the texts are similar, i did not configure the dataloader to look at all the samples in a epoch.

# Usage
Requirements
- ftfy version: 6.2.3
- spacy version: 3.7.5
- torchtext version: 0.18.0
- torch version: 2.3.0
- datasets version: 2.20.0
- wandb (Optional)

Example train scripts. ([script file](./experiments.sh))
```bash
# Base Line Model (ex-01)
python main.py --save_root "./model-store/ex01" --print_interval 0.2 \
    --n_layers 8 --epoch 512 --context_size 128 \
    --wb_flag --wb_project 'gpt' --wb_notes "base model" --wb_tag "base"

# Increase weights on special tokens (ex-02)
python main.py --save_root "./model-store/ex02" --print_interval 0.2 \
    --n_layers 8 --epoch 512 --context_size 128 --sp_weight 2.0 \
    --wb_flag --wb_project 'gpt' --wb_notes "Adjust class weights" --wb_tag "class_weight"
```

# Results

### Training Losses

![image](https://github.com/user-attachments/assets/bf76fc0c-4c8b-45c2-bf3d-8d323998100e)

### Text Generation
* Please check the [notebook](./TextGeneration.ipynb)
#### ex-01
* Input: `lily went to the store and bought a new toy.`
* Output: `lily went to the store and bought a new toy . she was so happy and could n't wait to play with it . but when she got home , she saw that her toy was missing . she looked everywhere but could n't find it . lily was sad and did n't know what to do . she asked her mom for help , but her mom did n't know how to fix it . lily was very sad and did n't know what to do . she went to her room and cried . her mom came in and asked what was wrong . lily told her about her missing toy . her mom said , " do n't worry , we can fix it together . " they worked together and lily was able to find her toy . she was so happy again . she hugged her mom was so happy again . lily was so happy again . she was so glad that she was n't have to be sad . from now . she was n't have to lose her toy . she was n't have to lose it . lily . she was so sad . she was so sad . her mom was so sad . her mom was so sad . her mom was so sad . her mom was so sad . her mom was so sad . she was so sad . her mom was so sad . her mom was so sad . she`
#### ex-02
* Input: `lily went to the store and bought a new toy.`
* Output: `lily went to the store and bought a new toy . she was so happy and played with it all day long . when it was time to go home , lily was tired but happy . she had a great day at the store and could n't wait to go back and play again . <eot>`

# To Do
* [x] Since the frequency of SOT token or EOT token is very low, put more weights on special tokens.
  - Code Update ([7783df0](https://github.com/tyui592/gpt_study/commit/7783df0c252967cfce097b3dfc2f02433c438888)) for this and did the experiment (`ex-02`) 
