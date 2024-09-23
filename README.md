# Beyond Final Products: Multi-Dimensional Essay Scoring Using Keystroke Logs and Deep Learning

This repository is the implementation of the KAES architecture.

## Package Requirements

Install below packages in your virtual environment before running the code.
- python==3.7.11
- tensorflow=2.0.0
- numpy=1.18.1
- nltk=3.4.5
- pandas=1.0.5
- scikit-learn=0.22.1

## Download GloVe

For prompt word embedding, we use the pretrained GloVe embedding.
- Go to [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) and download `glove.6B.50d.txt`.
- Put downloaded file in the `embeddings` directory.

## Run KAES Model
This bash script will run each model 5 times with different seeds ([9, 12, 42, 51, 86]).
- `bash ./train_KAES.sh`

Note that every run does not produce the same results due to the random elements.
