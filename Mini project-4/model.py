import torch
import torch.nn as nn
import re
import json

class ToxicityLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=6):
        super(ToxicityLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        # lstm_out shape: (batch, seq_len, hidden_dim * 2)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Take the average pool over sequence length or just the final state.
        # Average pooling handles variable sequence lengths better
        out = torch.mean(lstm_out, dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class CustomTokenizer:
    def __init__(self, num_words=10000, max_len=150):
        self.num_words = num_words
        self.max_len = max_len
        self.word_index = {"<PAD>": 0, "<UNK>": 1}
        self.index_word = {0: "<PAD>", 1: "<UNK>"}

    def fit_on_texts(self, texts):
        word_counts = {}
        for text in texts:
            words = self._clean_and_tokenize(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort words by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add to word index up to num_words limit
        for i, (word, _) in enumerate(sorted_words[:self.num_words - 2]):
            idx = i + 2
            self.word_index[word] = idx
            self.index_word[idx] = word

    def _clean_and_tokenize(self, text):
        text = str(text).lower()
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = []
            words = self._clean_and_tokenize(text)
            for word in words:
                seq.append(self.word_index.get(word, 1)) # Default to UNK
            sequences.append(seq)
        return sequences

    def pad_sequences(self, sequences):
        padded = []
        for seq in sequences:
            if len(seq) < self.max_len:
                # Pad with 0s at the end
                seq = seq + [0] * (self.max_len - len(seq))
            else:
                # Truncate
                seq = seq[:self.max_len]
            padded.append(seq)
        return padded

    def to_json(self):
        return json.dumps({
            "num_words": self.num_words,
            "max_len": self.max_len,
            "word_index": self.word_index
        })

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        tokenizer = cls(num_words=data["num_words"], max_len=data["max_len"])
        tokenizer.word_index = data["word_index"]
        tokenizer.index_word = {int(v): k for k, v in data["word_index"].items()}
        return tokenizer
