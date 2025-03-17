import os
import json
import pandas as pd

class Converter:
    """
    Converter class that searches a directory for JSON files, processes each one,
    and extracts two lists:
      - original_sentences: the raw text from each sentence (from the "text" key)
      - processed_sentences: each sentence converted word‐by‐word.
    
    Also, the class accepts a parameter "sentence_length" (an integer). Only sentences
    having no more than that number of words (counting only tokens with wtype=="word") are processed.
    If no JSON files are found in the directory, a FileNotFoundError is raised.
    """

    def __init__(self, directory: str, sentence_length: int = None):
        """
        :param directory: Directory path in which to look for JSON files.
        :param sentence_length: Maximum number of words (with wtype=="word") allowed in a sentence.
                                If a sentence is longer, it will be skipped.
        """
        self.directory = directory
        self.sentence_length = sentence_length
        self.original_sentences = []
        self.processed_sentences = []
        self.allowed_tags = ['1', '2', '3', 'masc', 'fem', 'sg', 'pl', 'nom', 'gen',
                             'dat', 'acc', 'abl', 'loc', 'pst', 'prs', 'fut']

    def convert(self):
        json_files = [f for f in os.listdir(self.directory) if f.endswith('.json')]
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in directory {self.directory}")

        for filename in json_files:
            file_path = os.path.join(self.directory, filename)
            with open(file_path, encoding='utf-8') as f:
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


if __name__ == "__main__":
    converter = Converter(directory="D:/ParaForge/data", sentence_length=10)
    try:
        originals, processed = converter.convert()
        pd.DataFrame(data={'original': originals, 'processed': processed}).to_csv("D:/ParaForge/data/output/output.csv", index=False)
    except FileNotFoundError as e:
        print(e)
