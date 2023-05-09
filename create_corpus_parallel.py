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
import concurrent.futures

parser = argparse.ArgumentParser(description="Create the text corpus")

parser.add_argument("--verbose", action="store_true", default=True,
                    help="verbose mode")
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
            if random.random() > 0.01:
                parsed_text += random.choice(phoneme_variant_dict[phoneme])
            else:
                parsed_text += random.choice(phoneme_chars)

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


def parse_book(book_path, book_parser=parse_as_phonemes_noisy):
    """
    Function to parse a single book.
    """
    encoding = "utf-8" if "utf-8" in book_path.lower() else "ascii"
    output_filepath = book_path[:book_path.rindex("/") + 1] +"_" +book_path[book_path.rindex("/") + 1:]
    
    try:
        with open(book_path, "r", encoding=encoding) as book_file:
            book_text = book_file.read()
            final_text, book_lines = book_parser(book_text)

            with open(
                output_filepath, "a+", encoding="utf-8"
            ) as output_file:
                output_file.write(final_text.strip() + "\n")
        print(f"Created {output_filepath}.")

        return output_filepath, book_lines
    
    except Exception as e:
        print(f"Error with {output_filepath}: {e}.")
        return "", 0


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

    with concurrent.futures.ThreadPoolExecutor() as executor:
        try:
            result = executor.map(parse_book, book_list)
        except KeyboardInterrupt:
            executor.cancel()
