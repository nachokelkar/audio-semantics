"""
Code to create phoneme sequences from
Gutenburg books data used in LibriSpeech
"""

import argparse
import os
import random
from multiprocessing import Process

from constants import delimiters, np_variant_ascii, p_variant_ascii
from g2p_en import G2p

parser = argparse.ArgumentParser(description="Create the text corpus")

parser.add_argument("--verbose", action="store_true", default=True,
                    help="verbose mode")
parser.add_argument("--sentence_limit", default=4000,
                    help="Limit sentence to a particular length")
parser.add_argument("--mode", default="p",
                    help="noisy phonemes (n) | phonemes (p) | characters (c)")
parser.add_argument("--noise_prob", default=0.01,
                    help="probability of noise (only for mode=n)")


# Step 1 : Create corpus with only characters
def parse_as_char(
        book_text: str,
        sentence_limit: int = 4000,
        *args, **kwargs
):
    """
    Parses the text of a book as character sequences,
    also returns the number of lines in the book
    """
    final_text = ""
    sent_len = 0

    for char in book_text:
        if char.isalpha() and sent_len <= sentence_limit:
            final_text += char.lower()
            sent_len += 1
        elif char in list(delimiters) or sent_len > sentence_limit:
            if final_text[-1] != "\n":
                final_text += "\n"
            sent_len = 0

    return final_text.strip()


# Step 2 : Create corpus with phonemes
def parse_as_phonemes(
        book_text: str,
        sentence_limit: int = 4000,
        phoneme_dict: dict = p_variant_ascii,
        *args, **kwargs
):
    """
    Parses the text of a book as phoneme sequences,
    also returns the number of lines in the book
    """
    g2p = kwargs['g2p']
    final_text = ""
    parsed_text = ""
    sent_len = 0

    text_as_phonemes = g2p(book_text)
    for phoneme in text_as_phonemes:
        phoneme = "".join([char for char in phoneme if char.isalpha()])
        if phoneme:
            parsed_text += phoneme_dict[phoneme]

    for char in parsed_text:
        if char.isalpha() and sent_len <= sentence_limit:
            final_text += char
            sent_len += 1
        elif char in list(delimiters) or sent_len > sentence_limit:
            if final_text[-1] != "\n":
                final_text += "\n"
            sent_len = 0

    return final_text.strip()


# Step 3 : Create corpus with phonemes, and with some variability
def parse_as_phonemes_noisy(
        book_text: str,
        sentence_limit: int = 4000,
        phoneme_variant_dict: dict = np_variant_ascii,
        *args, **kwargs
):
    """
    Parses the text of a book as phoneme sequences by adding some
    variants of each phoneme.
    Also returns the number of lines in the book.
    """
    g2p = kwargs["g2p"]
    phoneme_chars = [
        item for sublist in phoneme_variant_dict.keys() for item in sublist
    ]
    phoneme_chars.append("")

    final_text = ""
    parsed_text = ""
    sent_len = 0

    text_as_phonemes = g2p(book_text)
    for phoneme in text_as_phonemes:
        phoneme = "".join([char for char in phoneme if char.isalpha()])
        if phoneme:
            if random.random() > 0.01:
                parsed_text += random.choice(phoneme_variant_dict[phoneme])
            else:
                parsed_text += random.choice(phoneme_chars)

    for char in parsed_text:
        if char.isalpha() and sent_len <= sentence_limit:
            final_text += char
            sent_len += 1
        elif char in list(delimiters) or sent_len > sentence_limit:
            if final_text[-1] != "\n":
                final_text += "\n"
            sent_len = 0

    return final_text.strip()


def parse_book(
        book_path: str,
        book_parser: callable,
        mode: str,
        *args, **kwargs
):
    """
    Function to parse a single book.
    """
    if mode in ["n", "p"]:
        g2p = kwargs["g2p"]

    encoding = "utf-8" if "utf-8" in book_path.lower() else "ascii"
    output_filepath = os.getcwd() + \
        f"/data/{mode}_data/" + book_path[book_path.rindex("/") + 1:]

    try:
        with open(book_path, "r", encoding=encoding) as book_file:
            book_text = book_file.read()
            if mode in ["n", "p"]:
                final_text = book_parser(book_text, g2p=g2p)
            else:
                final_text = book_parser(book_text)

            with open(output_filepath, "w", encoding="utf-8") as output_file:
                output_file.write(final_text.strip() + "\n")

        return True

    except Exception as e:
        print(f"Error with {output_filepath}: {e}.")
        return False


def parse_all(
        book_list: str,
        n_procs: int,
        proc_id: int,
        mode="n"
):
    """
    Central parsing function that is passed to a process to take in a list of
    books and parse each based on a hash.
    """
    parser_dict = {
        "n": parse_as_phonemes_noisy,
        "p": parse_as_phonemes,
        "c": parse_as_char
    }
    g2p = G2p()

    print(f"Started P{proc_id}")
    for book_path in book_list:
        try:
            if (
                int(
                    book_path[book_path.rindex("/") + 1:book_path.rindex(".")]
                ) % n_procs == proc_id - 1
                and
                not os.path.isfile(
                    os.getcwd() + f"/data/{mode}_data/" +
                    book_path[book_path.rindex("/") + 1:]
                )
            ):
                parse_book(
                    book_path,
                    book_parser=parser_dict[mode],
                    mode=mode,
                    g2p=g2p
                )
        except ValueError:
            continue


if __name__ == "__main__":
    g2p = G2p()
    args = parser.parse_args()

    encodings = ["ascii", "utf-8"]
    book_folders = []

    # Obtain list of all book directories
    for encoding in encodings:
        filepath = f"data/LibriSpeech/books/{encoding}/"

        book_folders.extend([filepath + i for i in os.listdir(filepath)])

    book_list = []
    for book_folder in book_folders:
        book = os.listdir(book_folder)[0]
        book_path = book_folder + "/" + book
        book_list.append(book_path)

    n_procs = 4

    processes = [
        Process(
            target=parse_all,
            args=(book_list, n_procs, i+1, args.mode, )
        ) for i in range(n_procs)
    ]

    for process in processes:
        process.start()

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
