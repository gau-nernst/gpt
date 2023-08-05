# GPT

Download datasets

```bash
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

Train unigram tokenizer

```bash
python tokenizer.py train --input TinyStoriesV2-GPT4-train.txt --model_name tiny_stories_unigram --input_sentence_size 2000000
```

Tokenize train and test sets

```bash
python tokenizer.py tokenize --input TinyStoriesV2-GPT4-train.txt --model_name tiny_stories_unigram --save_path tiny_stories_train.bin
python tokenizer.py tokenize --input TinyStoriesV2-GPT4-valid.txt --model_name tiny_stories_unigram --save_path tiny_stories_valid.bin
```
