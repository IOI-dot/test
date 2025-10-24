import re
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = r"C:\Users\Omar\Downloads\archive (17)\emotion_sentimen_dataset.csv"
MAX_LEN = 100
BATCH_SIZE = 32
EMBED_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 1
LR = 0.0001
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def tokenize(text):
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

class Vocab:
    def __init__(self, texts, max_size=5000):
        from collections import Counter
        counter = Counter([t for txt in texts for t in tokenize(txt)])
        most_common = [w for w,_ in counter.most_common(max_size)]
        self.itos = ["<PAD>", "<UNK>"] + most_common
        self.stoi = {w:i for i,w in enumerate(self.itos)}
    def encode(self, tokens):
        ids = [self.stoi.get(t,1) for t in tokens][:MAX_LEN]
        return ids + [0]*(MAX_LEN-len(ids))

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, label2id):
        self.samples = [(vocab.encode(tokenize(t)), label2id[l]) for t,l in zip(texts,labels)]
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x,y = self.samples[i]
        return torch.tensor(x), torch.tensor(y)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS, dim_feedforward=128
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, NUM_LAYERS)
        self.fc = nn.Linear(EMBED_DIM, num_classes)
    def forward(self, x):
        x = self.embed(x)            # (batch, length, embed_dim)
        x = x.permute(1,0,2)         # (length, batch, embed_dim)
        out = self.encoder(x)        # transformer expects (L, B, E)
        out = out.mean(dim=0)        # average pooling over tokens → (B, E)
        return self.fc(out)

df = pd.read_csv(DATA_PATH)
texts, labels = df["text"].astype(str).tolist(), df["Emotion"].astype(str).tolist()
label2id = {l:i for i,l in enumerate(sorted(set(labels)))}
vocab = Vocab(texts)

X_train,X_val,y_train,y_val = train_test_split(texts,labels,test_size=0.2,stratify=labels)
train_ds = TextDataset(X_train,y_train,vocab,label2id)
val_ds   = TextDataset(X_val,y_val,vocab,label2id)
train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
val_loader   = DataLoader(val_ds,batch_size=BATCH_SIZE)

model = TransformerClassifier(len(vocab.itos), len(label2id)).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1,EPOCHS+1):
    model.train()
    for xb,yb in train_loader:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    model.eval()
    preds,true = [],[]
    with torch.no_grad():
        for xb,yb in val_loader:
            p = model(xb.to(DEVICE)).argmax(1).cpu()
            preds.extend(p); true.extend(yb)
    print(f"Epoch {epoch}: Val Acc = {accuracy_score(true,preds):.4f}")

torch.save(model.state_dict(), "encoder_model.pth")
