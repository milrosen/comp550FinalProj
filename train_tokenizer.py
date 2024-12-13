from tokenizers import SentencePieceBPETokenizer, SentencePieceUnigramTokenizer
from datasets import load_dataset, interleave_datasets
import sys

"af am ar as az be bg bn br bs ca cs cy da de el en eo es et eu fa ff fi fr fy ga gd gl gn gu ha he hi hi_rom hr ht hu hy id ig is it ja jv ka kk km kn ko ku ky la lg li ln lo lt lv mg mk ml mn mr ms my my_zaw ne nl no ns om or pa pl ps pt qu rm ro ru sa sc sd si sk sl so sq sr ss su sv sw ta ta_rom te te_rom th tl tn tr ug uk ur ur_rom uz vi wo xh yi yo zh-Hans zu"

"en ru vi ja de ro fr fi ko es zh-Hans it nl ar tr hi cs lt lv kk et ne si gu my"


def batch_iterator(ds, batch_size=1000):
    batch = []
    count = 0
    for example in ds:
        batch.append(example["text"])
        if len(batch) == batch_size:
            yield batch
            count += 1
            if count >= 75: break
            batch = []
    if batch:  # yield last batch
        yield batch

if __name__ == "__main__":
    tokenizer = SentencePieceUnigramTokenizer()
    langs = []
    monolingual = False
    prefix = None
    model = "unigram"
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if "--" in arg:
            match arg:
                case "--langs":
                    while i+1 < len(sys.argv) and "--" not in (arg := sys.argv[i+1]):
                        i += 1
                        langs.append(arg)
                case "--monolingual":
                    monolingual = True
                case "--model":
                    i += 1
                    if i >= len(sys.argv):
                        print("unspecified model")
                        exit()
                    model = sys.argv[i].lower()
                    if '--' in model:
                        print("unspecified model")
                        exit()
                    if model == "unigram":
                        tokenizer = SentencePieceUnigramTokenizer()
                    elif model == "bpe":
                        tokenizer = SentencePieceBPETokenizer()
                    else:
                        print(f"tokenizer model: {model} not supported yet :(")
                        
                case "--prefix":
                    i += 1
                    if i >= len(sys.argv):
                        print("unspecified prefix")
                        exit()
                    if "--" in sys.argv[i]:
                        print(f"cannot use command {sys.argv[i]} as a prefix")
                    prefix = sys.argv[i]
                
                    
        i += 1
    if len(langs) == 1: monolingual = True
        
    print(f"training {'monolingual' if monolingual else 'combined'} {model} tokenizer{'s' if monolingual else ''} for: {' '.join(langs)}")
                    
    datasets = [load_dataset(
        "cc100",
        split="train",
        lang=lang,
        streaming=True) 
    for lang in langs]
    
    if not monolingual:
        tokenizer = None
        if model == "unigram":
            tokenizer = SentencePieceUnigramTokenizer()
        elif model == "bpe":
            tokenizer = SentencePieceBPETokenizer()
        else:
            exit()
    
        dataset = interleave_datasets(datasets)

        if not prefix: prefix='-'.join(langs)
    
        tokenizer.train_from_iterator(
            batch_iterator(dataset),
            vocab_size=80_000,
            show_progress=True,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )

        tokenizer.save(f"tokenizers/tokenizer-cc-{prefix}.json")
        
    else:
        for dataset, lang in zip(datasets, langs):
            
            tokenizer = None
            if model == "unigram":
                tokenizer = SentencePieceUnigramTokenizer()
            elif model == "bpe":
                tokenizer = SentencePieceBPETokenizer()
            else:
                exit()
            
            tokenizer.train_from_iterator(
                batch_iterator(dataset, 5_000),
                vocab_size=40_000,
                show_progress=True,
                special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
            )
            prefix = lang
            
            tokenizer.save(f"tokenizers/tokenizer-cc-{prefix}.json")
            