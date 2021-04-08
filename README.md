# Topic Modelling in Online Chats

[download rest of files here](https://drive.google.com/drive/folders/13NQzZsDDtu-FVAZS1_BvspqrOIk9-lkQ?usp=sharing)

I initially started with one file and it grew so haven't sorted into folders yet but I've left a description of each file below. Will organise into folders once I have time to change the code which loads the files in each script.

## Notebooks
### `Chat_analsis_text.ipynb`
- Fitting LDA to aeternity chat
- Plotting LDA topics

### `Topic Analysis.ipynb`
- Loads and preprocesses data from Yahoo finance Chats
- Aggregates messages by time
- Fits LDA (with tuned hyperparameters based on coherence score)
- Visualises topics from LDA
- Gets topic embeddings using method outlined in [Top2Vec](https://arxiv.org/pdf/2008.09470.pdf)
- Fits [BERTopic](https://github.com/MaartenGr/BERTopic) model to data
- Plotting topic embeddings
- Original functions for off-topic score

### `Topic deviation measures.ipynb`
- Loads topic/message embeddings, BERTopic model and message DF
- Calculates deviation scores
- Splits messages into blocks and calculates block-level deviation

## Scripts
### `block_dev.py`
Helper function for calculating deviations for plotly dashboard
### `dA.py`
Plotly Dashboard with graphs for the following:
- Topic deviation score prototype I for PayPal Chat
- Topic embedding comparisons
- Deviation from start prototype II
- Deviation from main plot
- Deviation scores per block

Just run this script to show the dashboard

### `get_embeddings.py`
Calculates BERT embeddings for yahoo finance chats

### `pred_topics.py`
Fits BERTopic model to yahoo finance chats.

### `preprocessing.py`
Code for preprocessing Yahoo finance chats

### `Saving.py`
Helper functions for saving and loading data

## Pickles
- **base_hourly_gram:** BERTopic Model fitted on hourly blocked messages with bi-grams and tri-grams
- **base_hourly_noGram:** BERTopic Model fitted on hourly blocked messages without bi-grams or tri-grams
- **full_embs.pkl:** 2d and 5d BERT embeddings for messages in yahoo finance chats
- **full_embs.pkl:** 768d BERT embeddings for messages in yahoo finance chats
- **hourly_embs.pkl:** 2d and 5d BERT embeddings for hourly blocked messages
- **pypl_embs.pkl:** BERT embeddings for paypal chat
- **pypl_t_vecs.pkl:** Topic vectors from BERTopic model fitted to just PayPal Chat
- **top_results.pkl:** predictions for each message in Yahoo finance chats using model for normal BERTopic model
- **top_vecs.pkl:** 2d and 5d topic vectors for normal BERTopic model
- **top_vecs4plot.pkl:** All topic vectors scaled down to 2d for plotting on the Plotly dashboard
- **topNg_results.pkl:** predictions for each message in Yahoo finance chats using model for BERTopic model with no n-grams n>1
- **topNg_vecs.pkl:** 2d and 5d topic vectors for BERTopic model with no n-grams n>1

## CSV files
- **full_corpus.csv:** Full Yahoo finance chat dataframe
- **lda_log.csv:** Log for LDA model selection results
- **topic_words_noGram.csv:** top words for each topic in BERTopic model with no n-grams n>1
- **topic_words_wGram.csv:** top words for each topic in normal BERTopic model
- **Yahoo folder** messages from each individual yahoo finance chat# Carbon_Index
