"""
Code to create SentencePiece compatible corpus
from Gutenburg books data used in LibriSpeech
"""

import os
from tqdm import tqdm

VERBOSE = True
CORPUS_FILENAME = os.getcwd() + "/data/gutenberg_no_spaces.txt"

if __name__ == "__main__":
    if os.path.exists(CORPUS_FILENAME):
        os.remove(CORPUS_FILENAME)
    fp = open(CORPUS_FILENAME, 'w+')
    fp.close()

    encodings = ["ascii", "utf-8"]
    max_length = 0

    for encoding in encodings:
        if VERBOSE:
            print(f"Encoding - {encoding}")

        filepath = os.getcwd() + f"/data/LibriSpeech/books/{encoding}/"

        book_list = os.listdir(filepath)
        if VERBOSE:
            book_list = tqdm(book_list)

        for book_folder in book_list:
            try:
                book = os.listdir(filepath + book_folder)[0]
                book_path = filepath + book_folder + "/" + book

                with open(book_path, "r", encoding=encoding) as book_file:
                    book_text = book_file.read()
                    book_letters = "".join(
                        i.lower() for i in book_text if i.isalpha()
                    )
                    if len(book_letters) > max_length:
                        max_length = len(book_letters)

                    with open(CORPUS_FILENAME, "a") as corpus_file:
                        corpus_file.write(book_letters)
                        corpus_file.write("\n")

            except Exception:
                if VERBOSE:
                    book_list.set_postfix({"last error": book_folder, "max length": max_length})
