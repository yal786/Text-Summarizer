"""
priyal.py - Text Summarizer (Extractive + Abstractive)

Features:
1. Extractive summarization (Sentence Transformers).
2. Abstractive summarization using OpenAI API with chunking + hierarchical summarization.
3. Abstractive summarization using local Mistral model via Hugging Face.
4. Evaluation utilities (ROUGE).

Requirements:
    pip install nltk sentence-transformers scikit-learn transformers torch accelerate requests openai rouge-score
"""

import os, time, requests, math
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer

# -----------------
# Setup NLTK
# -----------------
nltk.download("punkt_tab")

# -----------------
# Shared Utils
# -----------------
def clean_text(text):
    return text.replace("\r\n", "\n").strip()

def split_into_sentences(text):
    return sent_tokenize(text)

def chunk_sentences(sentences, max_chars=3000, overlap=200):
    """
    Splits text into chunks (with overlap) for long-doc summarization.
    """
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        if cur_len + len(s) + 1 > max_chars:
            chunks.append(" ".join(cur))
            # add overlap
            overlap_text = ""
            while cur and len(overlap_text) < overlap:
                overlap_text = cur.pop() + " " + overlap_text
            cur = [overlap_text.strip()] if overlap_text.strip() else []
            cur_len = sum(len(x) for x in cur)
        cur.append(s)
        cur_len += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# -----------------
# 1. Extractive Summarizer
# -----------------
_model = None
def extractive_summary(text, k=5):
    """
    Picks top-k most representative sentences using embeddings.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = split_into_sentences(clean_text(text))
    if len(sentences) <= k:
        return " ".join(sentences)
    embeds = _model.encode(sentences, convert_to_numpy=True)
    doc_embed = embeds.mean(axis=0)
    sims = (embeds @ doc_embed) / (np.linalg.norm(embeds, axis=1) * np.linalg.norm(doc_embed) + 1e-8)
    top_idx = np.argsort(-sims)[:k]
    top_idx.sort()
    return " ".join([sentences[i] for i in top_idx])

# -----------------
# 2. Abstractive with OpenAI
# -----------------
OPENAI_KEY = "YOUR_API_KEY"

def openai_chat_summarize(text, model="gpt-3.5-turbo",
                          system="You are a helpful summarizer. Produce a concise summary."):
    """
    Calls OpenAI Chat API to summarize text.
    """
    if not OPENAI_KEY:
        raise ValueError("Set OPENAI_API_KEY in your environment.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Summarize the following text:\n\n{text}"}
        ],
        "temperature": 0.2,
        "max_tokens": 400
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"].strip()

def summarize_long_text_openai(full_text):
    """
    Hierarchical summarization: chunk → summarize each → summarize summaries.
    """
    sents = split_into_sentences(full_text)
    chunks = chunk_sentences(sents, max_chars=3000, overlap=300)
    chunk_summaries = []
    for ch in chunks:
        summary = openai_chat_summarize(ch)
        chunk_summaries.append(summary)
        time.sleep(0.5)
    combined = "\n\n".join(chunk_summaries)
    final = openai_chat_summarize(combined,
                                  system="You are a concise summarizer. Create a final cohesive summary.")
    return final

# -----------------
# 3. Abstractive with Local Mistral
# -----------------
try:
    from transformers import pipeline
    mistral_pipe = pipeline("text-generation",
                            model="mistralai/Mistral-7B-Instruct-v0.2",
                            device_map="auto",
                            torch_dtype="auto")
except Exception as e:
    mistral_pipe = None

def mistral_summarize(text, max_new_tokens=200):
    """
    Local summarization with Mistral instruct model.
    Requires GPU + enough VRAM.
    """
    if mistral_pipe is None:
        raise RuntimeError("Mistral pipeline not available. Install transformers + torch and load the model.")
    prompt = f"Summarize the following article in clear bullet points:\n\n{text}\n\nSummary:"
    out = mistral_pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
    return out.split("Summary:")[-1].strip()

# -----------------
# 4. Evaluation (ROUGE)
# -----------------
def evaluate(reference, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    return scorer.score(reference, summary)

# -----------------
# Example usage
# -----------------
if __name__ == "__main__":
    sample_text = """Artificial intelligence (AI) is intelligence demonstrated by machines, 
    in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks 
    define the field as the study of "intelligent agents": any device that perceives its environment 
    and takes actions that maximize its chance of successfully achieving its goals."""
    
    print("\n--- Extractive Summary ---")
    print(extractive_summary(sample_text, k=2))

    if OPENAI_KEY:
        print("\n--- OpenAI Abstractive Summary ---")
        print(summarize_long_text_openai(sample_text))

    if mistral_pipe:
        print("\n--- Mistral Abstractive Summary ---")
        print(mistral_summarize(sample_text))
