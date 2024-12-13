import json
import tokenizers
import os
from transformers import PreTrainedTokenizerFast

class NoOverlapTokenizer():
    """combines tokenizers into a single pretrained tokenizer
    by calculating offsets"""
    
    def __init__(self, tokenizer_files, path_to_tokenizers, tokenizer_langs, vocab_size):
        # self.tokenizers = [PreTrainedTokenizerFast(tokenizer_file=file) for file in tokenizer_files]
        # self.offset_indicies = [a + b + c = 80_000]
        # for tokenizer in tokenizers

        self.tokenizers = []
        self.langs = tokenizer_langs
        offset = 0
        cutoff =  (vocab_size - 5) // len(tokenizer_files)
        self.cutoff = cutoff

        try:
            os.mkdir(f"{path_to_tokenizers}partial")
        except FileExistsError:
            pass

        for f in tokenizer_files:
            out_f = f"{path_to_tokenizers}partial/{cutoff}-{"-".join(tokenizer_langs)}-{f}"

            if os.path.isfile(out_f): continue

            with open(f"{path_to_tokenizers}{f}", 'r', encoding='utf-8') as file:
                tokenizer_json = json.load(file)
                tokenizer_json["model"]["vocab"] = tokenizer_json["model"]["vocab"][0:cutoff]
                json_object = json.dumps(tokenizer_json, indent=4)
                
                with open(out_f, 'wb') as outfile:
                    outfile.write(json_object.encode("utf-8"))
            
            offset += cutoff

        for f in tokenizer_files:
            f = f"{path_to_tokenizers}partial/{cutoff}-{"-".join(tokenizer_langs)}-{f}"      
            self.tokenizers.append(PreTrainedTokenizerFast(tokenizer_file=f))
    
    def __call__(self, string, lang):
        try:
            lang_index = self.langs.index(lang)
            tokenizer = self.tokenizers[lang_index]
        except IndexError:
            print(f"lang: {lang} not found in {self.langs}, be sure you set the languages correctly when you created the combined tokenizer")
        toks = tokenizer(string)

        toks["input_ids"] = [idx + (self.cutoff - 5) * lang_index if idx > 4 else idx for idx in toks["input_ids"]]
        return toks
        
              
if __name__ == "__main__":
    nooverlap = NoOverlapTokenizer(['tokenizer-cc-en.json', 'tokenizer-cc-de.json', 'tokenizer-cc-vi.json'], "./tokenizers/", ["en", "de", "vi"], 80_000)

    # english space
    print(nooverlap("\u2581", "en"))
    # german space
    print(nooverlap("\u2581", "de"))