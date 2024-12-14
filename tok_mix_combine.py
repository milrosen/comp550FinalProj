import json
import os
from transformers import PreTrainedTokenizerFast

class TokMixTokenizer():
    """combines tokenizers at the token-level, merging units based on average probability"""
    
    def __init__(self, tokenizer_files, path_to_tokenizers, tokenizer_langs, vocab_size):
        self.tokenizers = []
        self.langs = tokenizer_langs

        try:
            os.mkdir(f"{path_to_tokenizers}partial")
        except FileExistsError:
            pass

        vocablularies = []
        vocabularies_words_only = []
        
        for f in tokenizer_files:
            out_f = f"{path_to_tokenizers}partial/tokmix-{"-".join(tokenizer_langs)}-{f}"

            if os.path.isfile(out_f): continue

            with open(f"{path_to_tokenizers}{f}", 'r', encoding='utf-8') as file:
                tokenizer_json = json.load(file)
                vocablularies.append(tokenizer_json["model"]["vocab"])
                vocabularies_words_only.append(
                    set(map(lambda x: x[0], tokenizer_json["model"]["vocab"]))
                )
        print("\u2581danke" in vocabularies_words_only[2])
        
        
    
    def __call__(self, string, lang, *args, **kwargs):
        pass
        
              
if __name__ == "__main__":
    nooverlap = TokMixTokenizer(['tokenizer-cc-en.json', 'tokenizer-cc-de.json', 'tokenizer-cc-vi.json'], "./tokenizers/", ["en", "de", "vi"], 80_000)

    # english space
    print(nooverlap("\u2581", "en"))
    # german space
    print(nooverlap("\u2581", "de"))
    