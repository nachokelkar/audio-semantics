"""
Code to create phoneme sequences from
Gutenburg books data used in LibriSpeech
"""

import os
from g2p_en import G2p
import argparse
from tqdm import tqdm
from re import split
from textwrap import wrap
import random

parser = argparse.ArgumentParser(description="Create the text corpus")

parser.add_argument("--verbose", action="store_true", default=True,
                    help="verbose mode")
parser.add_argument("--output",
                    help="output filepath")
parser.add_argument("--use_delimiters", default=".?!\n",
                   help="sentence delimiters (default=\".?!\\n\")")
parser.add_argument("--sentence_limit", default=4000,
                   help="Limit sentence to a particular length")
parser.add_argument("--n_lines", default=None,
                   help="Limit number of lines")
parser.add_argument("--mode", default="p",
                   help="phonemes with noise (n) | phonemes (p) | characters (c)")
parser.add_argument("--noise_prob", default=0.01,
                   help="probability of noise (only for mode=n)")


# Step 1 : Create corpus with only characters
def parse_as_char(book_text, line_limit=None, delimiters=".?!\n", sentence_limit=4000):
    """
    Parses the text of a book as character sequences,
    also returns the number of lines in the book
    """
    final_text = ""
    sent_len = 0
    n_lines = 0

    for char in book_text:
        if char.isalpha() and sent_len <= sentence_limit:
            final_text += char.lower()
            sent_len += 1
        elif char in list(delimiters) or sent_len > sentence_limit:
            if final_text[-1] != "\n":
                n_lines += 1
                if line_limit is None or n_lines <= line_limit:
                    final_text += "\n"
                else:
                    break
            sent_len = 0

    return final_text.strip(), n_lines


# Step 2 : Create corpus with phonemes
def parse_as_phonemes(book_text, line_limit=None, delimiters=".?!\n", sentence_limit=4000):
    """
    Parses the text of a book as phoneme sequences,
    also returns the number of lines in the book
    """
    global g2p
    final_text = ""
    sent_len = 0
    n_lines = 0

    parsed_text = "".join(g2p(book_text))

    for char in parsed_text:
        if char.isalpha() and sent_len <= sentence_limit:
            final_text += char.lower()
            sent_len += 1
        elif char in list(delimiters) or sent_len > sentence_limit:
            if final_text[-1] != "\n":
                n_lines += 1
                if line_limit is None or n_lines <= line_limit:
                    final_text += "\n"
                else:
                    break
            sent_len = 0

    return final_text.strip(), n_lines


# Step 3 : Create corpus with phonemes, and with some variability
def parse_as_phonemes_noisy(book_text, line_limit=None, delimiters=".?!\n", sentence_limit=4000):
    """
    Parses the text of a book as phoneme sequences by adding some
    variants of each phoneme.
    Also returns the number of lines in the book.
    """
    global g2p
    global args

    phoneme_variant_dict = {
        "AA": ["A", "a", "ā"],
        "AE": ["Ä", "ä"],
        "AH": ["ʌ", "ɐ"],
        "AO": ["ɒ", "ɔ"],
        "AW": ["Ą", "ą"],
        "AY": ["Æ", "æ"],
        "B": ["B", "b"],
        "CH": ["Ć", "ć", "Č", "č"],
        "D": ["D", "ɖ"],
        "DH": ["ð", "d"],
        "EH": ["ɛ", "e"],
        "ER": ["ɜ", "ɝ", "ɚ"],
        "EY": ["É", "é", "E"],
        "F": ["F", "f"],
        "G": ["G", "g", "ɠ"],
        "HH": ["H", "h"],
        "IH": ["i", "ɨ"],
        "IY": ["I", "ı"],
        "JH": ["J", "j", "ʄ", "ʝ"],
        "K": ["K", "k"],
        "L": ["L", "l"],
        "M": ["M", "m"],
        "N": ["N", "n"],
        "NG": ["Ń", "ń"],
        "OW": ["O", "o"],
        "OY": ["ɶ", "Œ", "ø"],
        "P": ["P", "p"],
        "R": ["R", "r", "ʁ"],
        "S": ["S", "s"],
        "SH": ["Ś", "ś", "ʃ"],
        "T": ["T", "t"],
        "TH": ["ʈ", "θ"],
        "UH": ["U", "u"],
        "UW": ["Ó", "ó"],
        "V": ["V", "v"],
        "W": ["W", "w", "Ł", "ł"],
        "Y": ["Y", "y"],
        "Z": ["Z", "z"],
        "ZH": ["Ż", "ż", "ʑ", "ʒ"]
    }
    phoneme_chars = [item for sublist in phoneme_variant_dict.keys() for item in sublist]
    phoneme_chars.append("")
    
    final_text = ""
    parsed_text = ""
    sent_len = 0
    n_lines = 0

    text_as_phonemes = g2p(book_text)
    for phoneme in text_as_phonemes:
        phoneme = "".join([char for char in phoneme if char.isalpha()])
        if phoneme:
            if random.random() > float(args.noise_prob):
                parsed_text += random.choice(phoneme_variant_dict[phoneme])
            else:
                parsed_text += random.choice(phoneme_chars)

    for char in parsed_text:
        if char.isalpha() and sent_len <= sentence_limit:
            final_text += char
            sent_len += 1
        elif char in list(delimiters) or sent_len > sentence_limit:
            if final_text[-1] != "\n":
                n_lines += 1
                if line_limit is None or n_lines <= line_limit:
                    final_text += "\n"
                else:
                    break
            sent_len = 0

    return final_text.strip(), n_lines


if __name__ == "__main__":
    g2p = G2p()
    args = parser.parse_args()

    VERBOSE = args.verbose

    # Store output filename
    CORPUS_FILENAME = os.getcwd() + f"/data/gutenberg.txt"
    if args.output is not None:
        CORPUS_FILENAME = args.output
        
    # Remove file if exists
    if os.path.exists(CORPUS_FILENAME):
        os.remove(CORPUS_FILENAME)

    encodings = ["ascii", "utf-8"]
    book_list = []
    
    # Obtain list of all book directories
    for encoding in encodings:
        if VERBOSE:
            print(f"Encoding - {encoding}")

        filepath = os.getcwd() + f"/data/LibriSpeech/books/{encoding}/"

        book_list.extend([filepath + i for i in os.listdir(filepath)])

    if VERBOSE:
        book_list = tqdm(book_list)

    parsers = {
        "c": parse_as_char,
        "p": parse_as_phonemes,
        "n": parse_as_phonemes_noisy
    }
    book_parser = parsers[args.mode]
    n_lines = 0
    lines_parsed = 0
    errors = 0

    for book_folder in book_list:
        try:
            book = os.listdir(book_folder)[0]
            book_path = book_folder + "/" + book
            
            if args.n_lines is None or n_lines < int(args.n_lines):
                with open(book_path, "r", encoding=encoding) as book_file:
                    book_text = book_file.read()
                    final_text, book_lines = book_parser(book_text)

                    with open(
                        CORPUS_FILENAME, "a+", encoding="utf-8"
                    ) as corpus_file:
                        if args.n_lines is not None and n_lines + book_lines > int(args.n_lines):
                            final_text = "\n".join(final_text.split("\n")[:n_lines + book_lines])
                        corpus_file.write(final_text.strip() + "\n")

                    n_lines += book_lines

        except Exception as e:
            if VERBOSE:
                errors += 1
                book_list.set_postfix({"errors": errors, "ECODE": e})

    if VERBOSE:
        print(f"Created file {CORPUS_FILENAME} with {n_lines} lines")
