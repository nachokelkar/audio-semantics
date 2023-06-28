from tokenizers import BertWordPieceTokenizer

special_tokens = [
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]

# if you want to train the tokenizer on both sets
# files = ["train.txt", "test.txt"]
# training the tokenizer on the training set
files = ["data/level_wise/level0/corpus_original.txt"]

# 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 60000

# initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    strip_accents=False,
    lowercase=False,
)
# train the tokenizer
tokenizer.train(
    files,
    vocab_size=vocab_size,
    min_frequency=0,
    show_progress=True,
    special_tokens=special_tokens
)

# Save the files
tokenizer.save_model("./", "lvl0-wp-tokenizer")
