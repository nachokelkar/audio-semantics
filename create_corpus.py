"""
Code to create SentencePiece compatible corpus
from Gutenburg books data used in LibriSpeech
"""

import os

VERBOSE = False
CORPUS_FILENAME = "gutenberg_no_spaces.txt"

if __name__ == "__main__":
    os.remove(CORPUS_FILENAME)
    fp = open(CORPUS_FILENAME, 'w+')
    fp.close()

    encodings = ["ascii", "utf-8"]

    for encoding in encodings:
        filepath = os.getcwd() + f"LibriSpeech/books/{encoding}/"

        book_list = os.listdir(filepath)

        for book_folder in book_list:
            try:
                book = os.listdir(filepath + book_folder)[0]
                book_path = filepath + book_folder + "/" + book

                with open(book_path, "r", encoding=encoding) as book_file:
                    book_text = book_file.read()

                    with open(CORPUS_FILENAME, "a") as corpus_file:
                        corpus_file.write(
                            "".join(
                                i.lower() for i in book_text if i.isalpha()
                            )
                        )
            except Exception:
                if VERBOSE:
                    print(f"at {encoding}: {book_folder}")
