"""
Code to create SentencePiece compatible corpus
from Gutenburg books data used in LibriSpeech
"""

import os
import argparse
from tqdm import tqdm
from re import split
from textwrap import wrap

parser = argparse.ArgumentParser(description="Create the text corpus")

parser.add_argument("--verbose", action="store_true", default=True,
                    help="verbose mode")
parser.add_argument("--output",
                    help="output filepath")
parser.add_argument("--spaces", action="store_true", default=False,
                    help="spaces or not")
parser.add_argument("--new_lines", action="store_true", default=False,
                    help="new lines between books or not")
# parser.add_argument("--use_delimiters", default=".|\?|\!|\n",
parser.add_argument("--use_delimiters", default=".?!\n",
                   help="split sentences with regex delimiter (default=\.|\?|\!|\n)")
parser.add_argument("--sentence_limit", default=4000,
                   help="Limit sentence to a particular length")
parser.add_argument("--n_lines", default=None,
                   help="Limit number of lines")


if __name__ == "__main__":
    args = parser.parse_args()

    VERBOSE = args.verbose

    CORPUS_FILENAME = os.getcwd() + f"/data/gutenberg.txt"
    if args.output is not None:
        CORPUS_FILENAME = args.output
    if os.path.exists(CORPUS_FILENAME):
        os.remove(CORPUS_FILENAME)
    fp = open(CORPUS_FILENAME, 'w+')
    fp.close()

    encodings = ["ascii", "utf-8"]
    max_length = 0

    book_list = []

    for encoding in encodings:
        if VERBOSE:
            print(f"Encoding - {encoding}")

        filepath = os.getcwd() + f"/data/LibriSpeech/books/{encoding}/"

        book_list.extend([filepath + i for i in os.listdir(filepath)])
        
    if VERBOSE:
        book_list = tqdm(book_list)
        
    n_lines = 0

    for book_folder in book_list:
        try:
            book = os.listdir(book_folder)[0]
            book_path = book_folder + "/" + book

            with open(book_path, "r", encoding=encoding) as book_file:
                if args.n_lines is not None and n_lines <= int(args.n_lines):
                    book_text = book_file.read()

                    final_text = ""
                    sent_len = 0

                    for char in book_text:
                        if char.isalpha() and sent_len <= args.sentence_limit:
                            final_text += char.lower()
                            sent_len += 1
                        elif char in list(args.use_delimiters) or sent_len > args.sentence_limit:
                            if final_text[-1] != "\n":
                                n_lines += 1
                                if args.n_lines is not None and n_lines <= int(args.n_lines):
                                    final_text += "\n"
                                else:
                                    break
                            sent_len = 0

                    with open(
                        CORPUS_FILENAME, "a", encoding="utf-8"
                    ) as corpus_file:
                            corpus_file.write(final_text.strip() + "\n")
                        # if args.new_lines:
                        #     corpus_file.write("\n")

        except Exception:
            if VERBOSE:
                book_list.set_postfix({"last error": book_folder})

    if VERBOSE:
        print(f"Created file {CORPUS_FILENAME}")

    # print("Max length :", max_length)
