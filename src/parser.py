import os
import json
from pathlib import Path
from typing import List, Tuple

def parse_json_file(file_path: Path) -> Tuple[List[str], List[str]]:
    """
    Parse a single JSON file and return both original and processed sentences.
    The processed sentences will be fed into the model to generate translations,
    while the original sentences will be paired with the generated translations
    in the final output.
    
    Args:
        file_path: Path to the JSON file to process
        
    Returns:
        Tuple of (original_sentences, processed_sentences)
    """
    converter = Converter(file_path, 
                          sentence_length=7)
    return converter.convert()

class Converter:
    """
    Converter class that processes a JSON file and extracts two lists:
      - original_sentences: the raw text from each sentence (from the "text" key)
      - processed_sentences: each sentence converted word‐by‐word.
    """

    def __init__(self, file_path: Path, sentence_length: int = None):
        """
        :param file_path: Path to the JSON file to process
        :param sentence_length: Maximum number of words (with wtype=="word") allowed in a sentence.
                                If a sentence is longer, it will be skipped.
        """
        self.file_path = file_path
        self.sentence_length = sentence_length
        self.original_sentences = []
        self.processed_sentences = []
        self.allowed_tags = ['1', '2', '3', 'masc', 'fem', 'sg', 'pl', 'nom', 'gen',
                             'dat', 'acc', 'abl', 'loc', 'pst', 'prs', 'fut']

    def convert(self) -> Tuple[List[str], List[str]]:
        """
        Convert the JSON file and return original and processed sentences.
        
        Returns:
            Tuple of (original_sentences, processed_sentences)
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"File {self.file_path} does not exist")

        with open(self.file_path, encoding='utf-8') as f:
            data = json.load(f)

        sentences = data.get("sentences", [])
        for sentence in sentences:
            words = sentence.get("words", [])
            word_count = sum(1 for word in words if word.get("wtype") == "word")
            if self.sentence_length is not None and word_count > self.sentence_length:
                continue

            original_text = sentence.get("text", "")
            self.original_sentences.append(original_text)

            processed_words = []
            for word in words:
                if word.get("wtype") != "word":
                    continue

                if "ana" in word and word["ana"]:
                    analysis = word["ana"][0]

                    trans_ru = analysis.get("trans_ru", "").strip()
                    trans_ru2 = analysis.get("trans_ru2", "").strip() if "trans_ru2" in analysis else ""
                    trans_parts = []
                    if trans_ru:
                        trans_parts.append(trans_ru)
                    if trans_ru2:
                        trans_parts.append(trans_ru2)
                    combined_trans = ", ".join(trans_parts)

                    tokens = [t.strip() for t in combined_trans.split(",") if t.strip()]
                    if tokens:
                        if len(tokens) > 1:
                            result_translation = tokens[0] + "[" + ", ".join(tokens[1:]) + "]"
                        else:
                            result_translation = tokens[0]
                    else:
                        result_translation = ""

                    collected_tags = set()
                    for key in ["gr.number", "gr.case", "gr.v_form", "gr.poss", "gloss"]:
                        if key in analysis:
                            val = analysis[key]

                            if isinstance(val, list):
                                for item in val:
                                    if item in self.allowed_tags:
                                        collected_tags.add(item)

                            elif isinstance(val, str):
                                for item in val.split(","):
                                    item = item.strip()
                                    if item in self.allowed_tags:
                                        collected_tags.add(item)

                    if collected_tags:
                        sorted_tags = sorted(collected_tags, key=lambda x: self.allowed_tags.index(x))
                        tags_str = "<" + ",".join(sorted_tags) + ">"
                    else:
                        tags_str = ""

                    processed_word = result_translation + tags_str
                    processed_words.append(processed_word)
                else:
                    processed_words.append(word.get("wf", ""))

            processed_sentence = " ".join(processed_words)
            self.processed_sentences.append(processed_sentence)

        return self.original_sentences, self.processed_sentences