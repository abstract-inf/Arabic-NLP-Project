## in the Question_asnwering code part
**Issue summary**
- During Word2Vec experiments, training and evaluation produced `nan` similarities.
- Root cause: cosine similarity is undefined for zero vectors. Word2Vec was returning all‑zero embeddings for many questions/answers because tokens were out‑of‑vocabulary (OOV), so `1 - cosine(p, t)` became `nan`. See Question_Answering.ipynb and Question_Answering.ipynb.

**How we fixed it**
1) Made cosine similarity safe by skipping/handling zero‑norm vectors so it never returns `nan`. See Question_Answering.ipynb.  
2) Reduced OOV by setting `min_count=1` in Word2Vec so rare Arabic tokens are included, lowering the number of zero vectors. See Question_Answering.ipynb.

**What this code does (in short)**
- Embedders: TF‑IDF, Word2Vec, FastText, SentenceTransformer convert questions and answers into fixed‑size vectors. See Question_Answering.ipynb.
- Models: RNN/LSTM/GRU/Transformer take question embeddings and learn to predict answer embeddings. See Question_Answering.ipynb.
- Training pipeline: trains a model with MSE loss and evaluates using cosine similarity between predicted and gold answer embeddings. See Question_Answering.ipynb.
- Experiment runner: loops over embedder + model combinations, trains each, and records test similarity and validation stats. See Question_Answering.ipynb.