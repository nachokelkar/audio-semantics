"""
Code to create SentencePiece compatible corpus
from Gutenburg books data used in LibriSpeech
"""

import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Create the text corpus")

parser.add_argument("--verbose", action="store_true", default=True,
                    help="verbose mode")
parser.add_argument("--output",
                    help="output filepath")
parser.add_argument("--n_books", default=None,
                    help="number of books to include in corpus")
parser.add_argument("--spaces", action="store_true", default=False,
                    help="spaces or not")


if __name__ == "__main__":
    args = parser.parse_args()

    VERBOSE = args.verbose

    n_books = -1
    if args.n_books is not None:
        n_books = int(args.n_books)

    CORPUS_FILENAME = os.getcwd() + f"/data/gns_{n_books}.txt"
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

    if n_books >= 1:
        book_list = book_list[:n_books]

    if VERBOSE:
        book_list = tqdm(book_list)

    for book_folder in book_list:
        try:
            book = os.listdir(book_folder)[0]
            book_path = book_folder + "/" + book

            with open(book_path, "r", encoding=encoding) as book_file:
                book_text = book_file.read()

                join_char = " " if args.spaces else ""
                book_letters = join_char.join(
                    i.lower() for i in book_text if i.isalpha()
                )
                if len(book_letters) > max_length:
                    max_length = len(book_letters)

                with open(CORPUS_FILENAME, "a") as corpus_file:
                    corpus_file.write(book_letters)
                    corpus_file.write("\n")

        except Exception:
            if VERBOSE:
                book_list.set_postfix({"last error": book_folder})
    
    if VERBOSE:
        print(f"Created file {CORPUS_FILENAME}")

    print("Max length :", max_length)