import json
import os
from transformers import PreTrainedTokenizerFast

class TokMixTokenizer():
    """combines tokenizers at the token-level, merging units based on average probability"""
    
    def __init__(self, tokenizer_files, path_to_tokenizers, tokenizer_langs, vocab_size):
        try:
            os.mkdir(f"{path_to_tokenizers}tokmix")
        except FileExistsError:
            pass

        vocablularies = []
        vocabularies_words_only = []

        out_f = f"{path_to_tokenizers}tokmix/{"-".join(tokenizer_langs)}-tokenizer-cc.json"
        if os.path.isfile(out_f): 
            with open(out_f, encoding='utf-8'):
                self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=out_f)
                return None

        
        for f in tokenizer_files:

            with open(f"{path_to_tokenizers}{f}", 'r', encoding='utf-8') as file:
                tokenizer_json = json.load(file)
                vocablularies.append(tokenizer_json["model"]["vocab"])
                vocabularies_words_only.append(
                    dict(map(lambda x: (x[1][0], x[0]), enumerate(tokenizer_json["model"]["vocab"])))
                )
        
        closed_list_words = set()
        count = 0
        langs_step = [5] * len(vocablularies)
        out_vocabulary = []
        while count <= vocab_size:
            for i, step in enumerate(langs_step):
                wd, val = vocablularies[i][step]
                while wd in closed_list_words:
                    langs_step[i] += 1
                    wd, val = vocablularies[i][langs_step[i]]
                    continue
                closed_list_words.add(wd)
                langs_step[i] += 1
                count += 1

                duplicate_count = 1
                avg_likelyhood = val
                for j in range(len(vocablularies)):
                    if i == j: continue
                    if wd in vocabularies_words_only[i]:
                        duplicate_count += 1
                        duplicate_wd_idx = vocabularies_words_only[i][wd]
                        avg_likelyhood += vocablularies[i][duplicate_wd_idx][1]
                out_vocabulary.append((wd, avg_likelyhood/duplicate_count))
        out_vocabulary = vocablularies[0][0:4] + out_vocabulary

        with open(f"{path_to_tokenizers}{tokenizer_files[0]}", 'r', encoding='utf-8') as file:
            tokenizer_json = json.load(file)
            tokenizer_json["model"]["vocab"] = out_vocabulary
            json_object = json.dumps(tokenizer_json, indent=4)
            with open(out_f, 'wb') as outfile:
                outfile.write(json_object.encode("utf-8"))
        
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=out_f)


        
        
    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)
        
              
if __name__ == "__main__":
    tokmix = TokMixTokenizer(['tokenizer-cc-en.json', 'tokenizer-cc-fr.json', 'tokenizer-cc-vi.json'], "./tokenizers/", ["en", "fr", "vi"], 80_000)
    print(tokmix(["eeaaoo", "bonjour", "nguyen", "hello my name is Milton", "Bonjour, je m'appelle Milton"]))
    