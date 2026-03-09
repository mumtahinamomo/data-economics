# Data Inequality and Market Power: Bengali vs English Corpus Analysis

---

## How to Run

### 1. Install Dependencies
```
pip install tensorflow datasets numpy pandas matplotlib requests tqdm
```

### 2. Download Bengali Wikipedia Articles (500)
```
python webscrap.py
```

### 3. Train Bengali Model
```
python train_bengali.py --n 50 --seed 0
```

### 4. Run Full Experiment for Low Resource language (Bengali)
```
python run_bengali.py
```

### 5. Comparison Graph
```
python graph_bengali.py
```

---

## Analysis: Data Inequality Between Bengali and English

### Overview

This experiment examines how unequal access to training data creates measurable disparities in AI model performance. Using Bengali Wikipedia as a low-resource language corpus and English OpenWebText as a high-resource corpus, we train identical LSTM language models on varying amounts of data and compare how performance scales across both languages.

The central argument mirrors income inequality in economics: just as wealth concentration limits economic mobility for lower-income populations, data concentration among dominant languages limits AI performance for speakers of low-resource languages. Bengali, despite being the 7th most spoken language in the world with over 230 million native speakers, remains trapped in a state of "data poverty."

---

### Corpus Statistics

| Metric | Bengali | English |
|---|---|---|
| Wikipedia articles (total) | ~158,000 | ~6,800,000 |
| Corpus size (this experiment) | 500 articles | 500 articles |
| Avg. article length (words) | ~491 | ~1,200+ |
| Total tokens (sample) | ~245,000 | ~600,000+ |
| Language resource level | Low-resource | High-resource |
| Wikipedia size ratio | 1x (baseline) | 43x larger |

English Wikipedia is approximately **43 times larger** than Bengali Wikipedia. This structural imbalance means that even when we train on the same number of documents, English provides far more tokens, vocabulary coverage, and linguistic diversity for the model to learn from.

---

### Experimental Results

We trained an LSTM next-word prediction model on both corpora, varying the number of training documents from 50 to 500 across 3 random seeds each. Performance was measured using Top 25 Categorical Accuracy on a validation set.

#### Key Findings

**Bengali (Low-Resource):**
- Accuracy range: 0.18 – 0.23
- Trend: Consistently declining as more documents are added
- Ceiling: ~0.23 at n=50, never exceeded regardless of data added
- Variance: High across seeds, indicating model instability

**English (High-Resource):**
- Accuracy range: 0.35 – 0.36
- Trend: Gradual decline with signs of potential stabilization
- Floor: ~0.35, never drops below Bengali's best performance
- Variance: Lower than Bengali, indicating more stable learning

#### The Inequality Gap

The most striking finding is that Bengali's best accuracy (0.23) never reaches English's worst accuracy (0.35). This gap persists regardless of how much Bengali data is added. 

---

### Interpretation: 

1. **Low data volume** leads to worse model performance
2. **Worse performance** leads to less adoption and usage
3. **Less usage** leads to less new data generated
4. **Less new data** leads to models stay poor and the cycle repeats

Bengali is caught in this  loop. With only 158,000 Wikipedia articles compared to English's 6.8 million, Bengali models are structurally disadvantaged. The declining accuracy curve we observe is evidence of the corpus hitting its quality ceiling.

This mirrors the Gini coefficient concept in income inequality: a small number of high-resource languages (English, Chinese, Spanish) hold the vast majority of training data "wealth," while the remaining 7,000+ languages share a tiny fraction. The result is compounding AI underperformance for billions of speakers.

---


## Files

| File | Description |
|---|---|
| `webscrap.py` | Downloads 500 Bengali Wikipedia articles |
| `train_bengali.py` | Trains LSTM model on Bengali corpus |
| `run_bengali.py` | Runs full experiment across all n and seeds |
| `graph_bengali.py` | Generates comparison graphs |
| `results_bengali.csv` | Bengali experiment results |
| `bengali_wiki_corpus.jsonl` | Raw Bengali article corpus |
| `graph_bengali_vs_english.png` | Final comparison chart |