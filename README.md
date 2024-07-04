# CS336 Spring 2024 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2024_assignment1_basics.pdf](./cs336_spring2024_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## An example generated prompt:
```
{0: ' In the a a small dog, meat. The sky was broken to wheels that all vis brightly to the toy. He fly to a very high in the window. They loved to play together. The house gave open bright, and Tim would turn warm inside. One day, Tim saw a small door. The door was shiny, and the light box would.\nTim went inside the house to look. He gazed at the flashlight and blinked at the flashlight. It smiled and had a big smile as Tim played with the flashlight on it. From that day on, Tim made sure everything would be fun to happen.\nOne day, Tim met a new friend Sue the house in her house. She had each red gears and small blue pants could shoot things too. Tim smiled and said, "How a veryFor my friend Tom."\n"I don\'t know, Tim. If you moved away, you have no hats and tails to crash," said Tim, licking. "You need to keep me safe to find each other and see where toys looks here before we play, solve cat and toy bugs. We can see if we start."\nTim was excited and understood. He remembered his eyes and thought fire were not a toy near the house. He wanted Tim to listen to her and make him happy. Tim remembered being a good friend. He also played and chew with Anna. One time, they played a new teddy bear together. Tim went to its mom and took off. He had a soft heart and a name and a nice pair named Jerry was playing with a dragon as a dragon dragon and many sharp ball.\nSadly, while they both talked and played with a long horn, Lily took off Zip\'s shiny toy. Max and Laser laughed, and it rocked even higher, and again. They became the best of friends and played every day.\n<|endoftext|>'}
```
on `https://wandb.ai/dhanvin_personal/cs336-assignment1/runs/sp_token_lr_1e-3_ramp_1k?nw=nwuserdhanvinm`

2.6k iterations

## Experiments
Logs for Training Runs:
```sh
https://wandb.ai/dhanvin_personal/cs336-assignment1/workspace?nw=nwuserdhanvinm
https://colab.research.google.com/drive/15ZoEOJlq5ZHs5IY1gPULwXNe_GFG9o09#scrollTo=jIwW7oxwyZJZ
```
* It's hard to tell from validation loss whether something has converged.
* What are good eval metrics for generated text?

## Instructions for running end-to-end (Tokenizer, Model Training, Inference):
1. BPE Tokenizer Training using Huggingface. Be sure to add <|endoftext|> as special token during training (see [tokenizer-training code](./cs336_basics/bpe_tokenizer/huggingface_bpe_trainer.py)). Ensure we have the right "dataset_name".
``` sh
python cs336_basics/bpe_tokenizer/huggingface_bpe_trainer.py
```
Check that we have an <|endoftext|> in the vocab.json file that's created.

2. Test encoding with a sample from the training data. Manually inspect the tokens with the vocab.json file. 
Configure and run:
``` sh
python -m cs336_basics.bpe_tokenizer.encoder_decoder
```
for *both the training and validation files* by editing the `dataset_name` in the script. 
NOTE: 
* By convention all vocab / merges from training lie in the training / validation dataset directory. Inspect `/data/TinyStoriesV2-GPT4` directory structure.
* You might need to manually delete the token files under the dataset directory e.g. `/data/TinyStoriesV2-GPT4-train-tokens.npy` to ensure they are regenerated by this script.

3. Upload the tokenized dir to Google Drive

4. Use Google Colab to train the model (A100 GPU)
``` sh
python cs336_basics/transformer/training_loop.py \
--name='cpu_test_wandb' --dataset_dir='/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment1-basics/data/TinyStoriesV2-GPT4' \
--checkpoint_path='/tmp/debug_checkpoints/' \
--training_batch_size=4 \
--total_train_tokens=327680000 \
--training_batch_size=32 \
--eval_batch_size=100 \
--lr_max=1e-4 \
--lr_min=1e-5 \
--lr_warmup_iters=300
```
Track performance in wandb. Ensure that validation and training curves are converging.

5. Download the trained model checkpoint

6. Run model inference to generate
```sh
python cs336_basics/transformer/inference.py \
--tokenizer_dir='/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment1-basics/data/TinyStoriesV2-GPT4/' \
--checkpoint_file '/Users/rajvimehta/Dhanvin-Code/cs336/spring2024-assignment1-basics/model_checkpoints/lr_5x-10-3-slower-ramp_checkpoint.pt' \
--max_tokens=500
```


NOTE: If our tokenization is done poorly, we must repeat everything again :/

## TODOs (Dhanvin)
1. Experiment with training → batch sizes, dropout, a little more on learning rate
2. Decoder - check runtime, quality etc. Play around with 
3. Experiment with architectural change → parallel
4. Fast decoder inference: Implement KV cacheing for decoding
5. Add Application metrics


## Doubts - Tokenizer
1. Unable to get the training to run efficiently with larger dataset of a GB (out of RAM, page-thrashing). Huggingface does it very effectively. However, the training vocab / merge-list seems to be at unicode char level as opposed to byte level.
2. Tokenization v/s feature-extraction: understanding the difference. Seems like tokenization is a loss-less feature-extraction relevant to NLP?


## Doubts - Model Archtitecture
1. What are Mixture of Expert LMs and (routing functions and sparse FFNs)
2. Inverted-bottleneck concept: Why does the FFN expand to 4x dim and then compress as opposed to compress to lower-dim and then expand?
3. RoPE v/s Absoulte-Learned v/s Sinusoid-Fixed: Why use learned position weights instead of fixed sinusoidal? What motivated fixed sinusoidal in the first place?
4. What is the intuition for residual connections? Why does pre-norm (having an uninterrupted residual path help learning stability)
5. What is energy-based models / diffusion / GANs / auto-regressive models?
6. Validation Loss computation: 
* Should I sample the entire validation set and split it into non overlapping sets of context-length (sliding window v/s chunking?)
* How to deal with dropout? Should I use model.eval()?


## Doubts - Loss Curves v/s Eval
1. How can I check during training that the decoder quality is good? What kinds of application metrics


## Setup

0. Set up a conda environment and install packages:

``` sh
conda create -n cs336_basics python=3.10 --yes
conda activate cs336_basics
pip install -e .'[test]'
```

1. Run unit tests:

``` sh
pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

2. Download the TinyStories data and a subsample of OpenWebText:

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

