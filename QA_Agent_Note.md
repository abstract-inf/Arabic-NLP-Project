Implemented. I created and populated all three new notebooks with full pipeline scaffolding, logging prints, tqdm-based progress, BLEU/ROUGE-L evaluation, per-model sections, and artifact exports.

Created notebooks:

question_answering_seq2sqe.ipynb
question_answering_transformers.ipynb
question_answering_rag.ipynb
What is now implemented:

Seq2seq notebook
Multi-dataset registry over raw + all CSVs in preprocessed datasets
Embedding strategies wired:
BoW
TF-IDF
Word2Vec Skip-gram
Word2Vec CBOW
fastText
BERT pooled embeddings
GPT pooled embeddings
Architectures wired:
RNN
LSTM
GRU
BiGRU
Full training/evaluation runners with epoch logs + tqdm bars
BLEU and ROUGE-L tracking per epoch
Best checkpoint saving per architecture with dataset + embedding provenance
Exports:
outputs/question_answering_seq2sqe_all_results.csv
outputs/question_answering_seq2sqe_best_models.csv
outputs/question_answering_seq2sqe_examples.csv
outputs/question_answering_seq2sqe_best_manifest.json
Loss and quality-vs-epoch plotting per architecture
Transformer notebook
Full fine-tuning engine for:
GPT
T5
Qwen
Separate model sections/cells (as requested), each with independent run flags and outputs
Three input/embedding strategy variants for training comparison:
plain_question
question_plus_category
instructional
BLEU and ROUGE-L epoch-level validation and final test summaries
Best checkpoint logic per model architecture with provenance
Combined comparison and per-model curve plotting
Exports:
outputs/question_answering_transformers_detailed.csv
outputs/question_answering_transformers_comparison.csv
outputs/question_answering_transformers_examples.csv
outputs/question_answering_transformers_best_manifest.json
RAG notebook
bge-m3 local embedding + NumPy cosine retrieval (top-K)
Exact Arabic RAG prompt template implemented
Loads best transformer checkpoint from notebook 2 manifest and uses it for generation
RAG evaluation with BLEU and ROUGE-L
Baseline-vs-RAG comparison table
Exports:
outputs/question_answering_rag_results.csv
outputs/question_answering_rag_examples.csv
outputs/question_answering_rag_comparison.csv
outputs/question_answering_global_summary.csv (global consolidation)
