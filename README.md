# Financial Sentiment Analysis with DistilBERT

Classifying sentences from financial news as **negative**, **neutral** or **positive** from an investor's
point of view, by fine-tuning a pre-trained Transformer (DistilBERT) with PyTorch and Hugging Face.

| Model | Accuracy | Macro-F1 |
|---|---|---|
| DistilBERT, frozen encoder (linear head only) | 71.6% | 0.45 |
| **DistilBERT, fully fine-tuned** | **94.0%** | **0.93** |

> Started as a lab of the NLP course at ENSIMAG (Grenoble INP).

## Dataset

[Financial PhraseBank v1.0](https://huggingface.co/datasets/takala/financial_phrasebank) (Malo et al., 2014):
sentences from financial news annotated by people with a finance background. I use the subset where
**at least 75% of annotators agree**: **3,453 sentences**, split 80% train / 20% test.

The classes are imbalanced (about 62% neutral, 26% positive, 12% negative): a model that always predicts
"neutral" already reaches about 62% accuracy. The main metric is therefore the **macro-F1**, which gives the
same weight to every class. The negative and positive classes are the ones that matter to an investor.

## Approach

1. **Baseline**: TF-IDF features + multinomial Naive Bayes.
2. **Tokenization analysis**: DistilBERT accepts 512 tokens, but 99% of the sentences are shorter than
   70 tokens (max 150), so the maximum length is reduced to **128** to save computation.
3. **Frozen encoder**: DistilBERT is used as a fixed feature extractor; only a linear classifier on the
   `[CLS]` token is trained (2,307 trainable parameters out of 66M).
4. **Full fine-tuning**: all 66M parameters are trained (3 epochs, learning rate 2e-5 with 10% warmup,
   weight decay 0.01, batch size 16, mixed precision on GPU).

## Results

**The frozen encoder is not enough.** Its 71.6% accuracy is misleading: the model predicts "neutral" almost
everywhere and **never correctly classifies a negative sentence**. Negative sentences are even predicted
*positive* more often than neutral: the pre-trained `[CLS]` vector seems to capture that "something
significant happened", but not its direction. A linear layer of 2,307 parameters cannot learn that.

**Fine-tuning fixes it**: the macro-F1 goes from **0.45 to 0.93**, with balanced scores on every class.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| neutral | 0.96 | 0.96 | 0.96 |
| positive | 0.92 | 0.90 | 0.91 |
| negative | 0.91 | 0.91 | 0.91 |

**Error analysis.** *"This year's profits quintuplicated with respect to the last year"* is predicted
**neutral** (0.77) instead of positive. "Quintuplicated" is a rare word, split into several sub-word tokens,
and the sentence uses none of the usual growth expressions ("rose", "increased"). With about 2,800 training
sentences, the model handles typical financial phrasing well but falls back to the majority class on
unfamiliar vocabulary.

## Run it

Open `tp_bert.ipynb` in Jupyter or [Google Colab](https://colab.research.google.com/) (a GPU is recommended)
and put `Sentences_75Agree.txt` from Financial PhraseBank in the same folder.

## Next steps

- Fix the TF-IDF baseline (fit the vectorizer on the training split only) and tune it on a validation split
- Seeded, stratified splits and k-fold cross-validation for more reliable scores
- Export to ONNX with INT8 quantization and measure the CPU speed-up
- Serve the model with FastAPI and Docker
- Compare with zero-shot prompting of an LLM (accuracy vs. cost and latency)

## Tech stack

Python · PyTorch · Hugging Face Transformers (DistilBERT, Trainer) · scikit-learn · pandas ·
Matplotlib / Seaborn · TensorBoard

## Data licence

Financial PhraseBank is released under CC BY-NC-SA 3.0 (non-commercial use).
