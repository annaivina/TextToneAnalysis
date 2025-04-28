import re
import string
from collections import Counter

class TextVectoriser():
    def __init__(self, max_len: int, min_freq: int, keep_emojis=False, remove_punctuation = False, is_Transformer=False):

        """
            TextVectoriser build the vocabu;laru form the provided input. 
            The output of the class is the tokenised inputs.

            Parameters:
                max_len: This is the maximum length of the input
                min_freq: How often word needs to be repeated to be put into voc.
                keep_emojis: You can also identify emogits in your dataset ans tokenize it
                remove_punctuation: removes any punctuation signs form the input. Note this will not touch emojis because we preprocess them early
                is_Transformer: this is for the inputs for the transfoer network. The input is main sentance + parent sentance 
                saparated by the <SEP> 

        """
        self.max_len = max_len
        self.min_freq = min_freq
        self.keep_emojis = keep_emojis
        self.remove_punctuation = remove_punctuation
        self.is_Transformer = is_Transformer
        self.vocab = None


    def process_text(self, text):
        text = text.lower()
        text = text.replace("&amp;","&")

        if self.is_Transformer:
            text = text.replace("<sep>", "SPECIALSEP")

        emoticon_pattern = re.compile(r'[:;=8][-~]?[)dDpPoO/(\\|]')
        emoticons_found = emoticon_pattern.findall(text)

        if self.keep_emojis:
             # Replace each emoticon with a placeholder
             for i, emo in enumerate(emoticons_found):
                 text = text.replace(emo, f'EMOTICON{i}')
        
        if self.remove_punctuation:
            text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
        else:
            text = re.sub(r"([.,!?\"“”():;])", r" \1 ", text)

        text = re.sub(r"[/\\]", " ", text)
        text = re.sub(r"\s+", " ", text)

        # 3️⃣ Put emoticons back
        if self.keep_emojis:
            for i, emo in enumerate(emoticons_found):
                text = text.replace(f'EMOTICON{i}', f' {emo} ')

        if self.is_Transformer:
            text = text.replace("SPECIALSEP", "<sep>")
        
        return text.strip()
    

    def build_vocab(self, texts):
        counter = Counter()
        for text in texts:
            tokens = self.tokinize(text)
            counter.update(tokens)
            
        vocab = {"<pad>": 0, "<unk>": 1}
        if self.is_Transformer:
            vocab['<sep>'] = 2
        for token, count in counter.items():
            if count >= self.min_freq:
                vocab[token] = len(vocab)
        self.vocab = vocab
        print(f"Vocab size: {len(vocab)}")
        return vocab
    
    def tokinize(self, text):
        return self.process_text(text).split()

    def vectorise(self, texts):
        tokens = self.tokinize(texts)
        token_ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]

        # Pad or truncate
        if len(token_ids) < self.max_len:
            token_ids += [self.vocab["<pad>"]] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[:self.max_len]

        return token_ids





