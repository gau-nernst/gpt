import argparse

import numpy as np
import sentencepiece as spm


if __name__ == "__main__":
    # https://github.com/google/sentencepiece/blob/master/doc/options.md
    parser = argparse.ArgumentParser()
    parser.add_argument("action")
    parser.add_argument("--input", required=True)
    parser.add_argument("--save_path")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--model_type", default="unigram")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--input_sentence_size", type=int, default=0)
    parser.add_argument("--max_sentence_length", type=int, default=4192)
    args = parser.parse_args()

    if args.action == "train":
        spm.SentencePieceTrainer.train(
            input=args.input,
            model_prefix=args.model_name,
            model_type=args.model_type,
            vocab_size=args.vocab_size,
            character_coverage=1.0,
            input_sentence_size=args.input_sentence_size,
            max_sentence_length=args.max_sentence_length,
            shuffle_input_sentence=(args.input_sentence_size > 0),
        )

    elif args.action == "tokenize":
        assert args.save_name is not None
        tokenizer = spm.SentencePieceProcessor(f"{args.model_name}.model")
        vocab_size = tokenizer.vocab_size()
        assert vocab_size < 2**15  # int16

        stories = open(args.input).read().split("<|endoftext|>")
        stories = [x for x in stories if len(x) >= 10]  # remove very short stories
        tokens_list = tokenizer.Encode(stories, add_bos=True)

        size = sum(len(x) for x in tokens_list)
        data = np.memmap(args.save_path, np.int16, "w+", 0, size)
        i = 0

        for tokens in tokens_list:
            data[i : i + len(tokens)] = tokens
            i += len(tokens)
        data.flush()
