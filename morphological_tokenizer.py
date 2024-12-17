import csv
import string
from nltk import word_tokenize
from tok_mix_combine import TokMixTokenizer
from transformers import PreTrainedTokenizerFast, AutoTokenizer
from datasets import load_dataset

class MorpholigicalGoldStandard():
    def __init__(self,path_to_tsv):
        derivational = {}
        self.morphtable = {}
        only_roots = set()
        
        with open(path_to_tsv, encoding="utf-8") as tsv:
            rd = csv.reader(tsv, delimiter="\t", quotechar="\"")
            for root, compound, _, _, morpheme, relation in rd:
                derivational[compound] = [root, relation == "suffix", morpheme]
                if len(root) <= 8:
                    only_roots.add(root)

        for word, [root, relation, morpheme] in derivational.items():
            if word in only_roots: only_roots.remove(word)

            if word not in self.morphtable:
                self.morphtable[word] = [[],[]]
            
            self.morphtable[word][relation].append(morpheme)
            closed = set()
            while root in derivational:
                if root in closed: break
                closed.add(root)
                
                root, relation, morpheme = derivational[root]
                self.morphtable[word][relation].append(morpheme)

            stem = word[sum(map(len, self.morphtable[word][0])):len(word)-sum(map(len, self.morphtable[word][1]))]
            
            prefixes, suffixes = self.morphtable[word]
            self.morphtable[word] = prefixes + [stem] + list(reversed(suffixes))

        
        for word in only_roots:
            self.morphtable[word] = [word]

    def gen_morpholigical_score(self, tokenizer, corpus, max_steps):

        errors = 0
        steps = 0
        for sentence in corpus:
            if steps >= max_steps: break
            if (isinstance(sentence, dict)):
                sentence = sentence["text"]
            sentence = sentence.translate(str.maketrans("", '', string.punctuation))
            words = word_tokenize(sentence)
            
            for word in words:
                if (steps+1) % 1_000 == 0:
                    print(f"\tfinished {steps} steps, error rate: {errors/steps}")

                steps += 1
                if word not in self.morphtable: continue
                splits = "-".join(self.morphtable[word])
                tokens = [tokenizer.decode(tk, skip_special_tokens=True) for tk in tokenizer(word).input_ids]
                if len(tokens) == 1: continue
                if tokens[0] == '': tokens = tokens[1:-1]
                tokens = '-'.join(tokens)

                t_i = 0
                s_i = 0

                while s_i < len(splits) and t_i < len(tokens):
                    s, t = splits[s_i], tokens[t_i]
                    if s == t:
                        s_i += 1
                        t_i += 1
                        continue
                    elif s == "-": s_i += 1
                    elif t == "-":

                        if steps <= 20: print("\t", splits, tokens)
                        errors += 1
                        t_i += 1
                    else:
                        s_i += 1
                        t_i += 1
        return errors / steps

if __name__ == "__main__":
    # mph = MorpholigicalGoldStandard("MorphyNet/eng/eng.derivational.v1.tsv")

    tokenizer_congfigs = [
        ["FacebookAI/xlm-clm-enfr-1024",["en", "fr", "es"]], # language embeddings, just en-f
        ["FacebookAI/xlm-mlm-17-1280",["en", "es", "de"]], # no language embeddings, 17 languages
        ["FacebookAI/xlm-roberta-large", ["ca", "cs", "de", "en", "fi", "fr", "hu", "it", "mn"]], #100 Langs
        ["facebook/bart-large",  ["ca", "cs", "de", "en", "fi", "fr", "hu", "it", "mn"]],  # BART
        ["bert-base-uncased", ["ca", "de", "en", "mn"]],
        ["bert-base-multilingual-uncased",["ca", "cs", "de", "en", "fi", "fr", "hu", "it", "mn"]],
        ["gpt2",["ca", "cs", "de", "en", "fi", "fr", "hu", "it", "mn"]],

        # "facebook/wmt19-ru-en", # Tokenizers don't share vocab, and, strangely enough, they decode in the /opposite/ language than they input. 
    ]

    tokenizers = [AutoTokenizer.from_pretrained(tk_name) for tk_name, _ in tokenizer_congfigs]

    tokenizers += [TokMixTokenizer(["tokenizer-cc-en.json", "tokenizer-cc-ru.json", "tokenizer-cc-fi.json"], "tokenizers/", ["en", "ru", "fi"], 80_000),
                   PreTrainedTokenizerFast(tokenizer_file="./tokenizers/tokenizer-cc-all.json")]
    tokenizer_congfigs += [["tokmix_en_ru_fi", ["en", "ru", "fi"]], 
                           ["unigram-all", ["en", "ru", "cs", "de", "fi", "fr", "hu", "it", "mn", "ca"]]]

    two2three = {"ca":"cat", "fr":"fra", "es":"spa", "en":"eng", "de":"deu", "it":"ita", "cs":"ces", 
                 "fi":"fin", "hu":"hun", "mn":"mon", "ru":"rus"}

    for tokenizer, [name, langs] in zip(tokenizers, tokenizer_congfigs):
        for lang in ["ca", "cs", "de", "en", "fi", "fr", "hu", "it", "mn", "ru"]:
            print(f"{name} in {lang}")
            dataset = load_dataset(
                "cc100",
                split="train",
                lang=lang,
                trust_remote_code=True,
                streaming=True) 

            print("\tbuilding morphtable")
            mph = MorpholigicalGoldStandard(f"MorphyNet/{two2three[lang]}/{two2three[lang]}.derivational.v1.tsv")
            print("\tcalculating score")
            mph.gen_morpholigical_score(tokenizer, dataset, 10_000)


    # mph.gen_morpholigical_score(nctk, txt.split("\n"), 10_000)

    # I want to test bert, bart, xlm-r, m-bart and m-bert tokenizers.

    # at least, along with any other models I find

