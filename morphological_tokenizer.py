import csv
import string
from nltk import word_tokenize
from tok_mix_combine import TokMixTokenizer

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

            sentence = sentence.translate(str.maketrans("", '', string.punctuation))
            words = word_tokenize(sentence)

            for word in words:
                if steps >= max_steps: break
                steps += 1
                if word not in self.morphtable: continue
                splits = "-".join(self.morphtable[word])
                tokens = [tokenizer.decode(tk) for tk in tokenizer(word).input_ids]
                if len(tokens) == 1: continue
                tokens = '-'.join(tokens)

                t_i = 0
                s_i = 0

                while s_i < len(splits) and t_i < len(tokens):
                    s, t = splits[s_i], tokens[t_i]
                    if s == t:
                        s_i += 1
                        t_i += 1
                        continue
                    if s == "-": s_i += 1
                    if t == "-":
                        errors += 1
                        t_i += 1
                print(splits, tokens, errors)
        return errors / steps

if __name__ == "__main__":
    
    txt = """The Investigating undesirable Phenomenology of Spirit , first published in 1807, is a work 
seen by Hegel as a necessary forepiece to his philosophical sys¬
tem (as later set forth in the Encyclopaedia of the Philosophical 
Sciences in Outline of 1817, 1827, anc * 1830), but it is meant to 
be a forepiece that can be dropped and discarded once the 
student, through deep immersion in its contents, has advanced 
through confusions and misunderstanding to the properly 
philosophical point of view. Its task is to run through, in a scien¬
tifically purged order, the stages in the mind’s necessary pro¬
gress from immediate sense-consciousness to the position of a 
scientific philosophy, showing thereby that this position is the 
only one that the mind can take, when it comes to the end of 
the intellectual and spiritual adventures described in the book. 
But this sort of history, he tells us in Encyclopaedia §25, necessarily 
had to drag in, more or less out of place and inadequately 
characterized, much that would afterwards be adequately set 
forth in the system, and it also had to bring in many motivating 
connections of which the adventuring mind was unaware, 
which explained why it passed from one phase of experience 
or action to another, and yet could not be set forth in the full 
manner which alone would render them intelligible. 

Hegel also, in preparing for republication of the work before 
his death in 1831, wrote a note which throws great light on 
his ultimate conception ofit. It was, he writes, a peculiar earlier 
work (eigentumlichefruhere Arbeit) which ought not to be revised, 
since it related to the time at which it was written, a time 
at which an abstract Absolute dominated philosophy. (See the 
final paragraph of the first section of Hoffmeister’s Appendix 
Zur Fes ts tel lung data Textes in the 1952 edition.) This note indi¬
cates that, while Hegel undoubtedly thought that the sequence 
of thought-phases described in the Phenomenology —phases ex¬
perienced by humanity in the past and recapitulated by Hegel 
in his own thought adventures up to and including his own ad¬
vance to the position of Science in about 1805—was a necessary """

    mph = MorpholigicalGoldStandard("MorphyNet/eng/eng.derivational.v1.tsv")
    nctk = TokMixTokenizer(["tokenizer-cc-en.json"], "tokenizers/", ["en"], 40_000)

    mph.gen_morpholigical_score(nctk, txt.split("\n"), 10_000)
