# Sequence learning with RNNs

An introduction to Recurrent Neural Networks (RNNs) for sequence modeling tasks. This repository contains educational notebooks that progress from basic RNN concepts to a full sentiment analysis pipeline using TensorFlow/Keras.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [alphabet_rnn.ipynb](notebooks/alphabet_rnn.ipynb) | **Start here.** A gentle introduction to RNNs using a simple next-character prediction task. Learn how RNNs maintain hidden state to process sequences. |
| [sentiment_analysis.ipynb](notebooks/sentiment_analysis.ipynb) | Full sentiment analysis pipeline on Twitter data. Covers tokenization, GloVe embeddings, Bidirectional GRU, and model evaluation with visualizations. |
| [sentement_analysis_activity.ipynb](notebooks/sentement_analysis_activity.ipynb) | **Activity:** Extend the baseline GRU model by adding convolutional layers to create a CNN-RNN hybrid architecture. Compare performance against the baseline. |

## Getting started

### Prerequisites

- [VS Code](https://code.visualstudio.com/)
- [Docker](https://www.docker.com/products/docker-desktop/)
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd RNNs
   ```

2. Open in VS Code:
   ```bash
   code .
   ```

3. When prompted, click **"Reopen in Container"** (or use the command palette: `Dev Containers: Reopen in Container`).

4. Wait for the container to build and install dependencies.

5. Open a notebook and run the cells.

### Data

The sentiment analysis notebooks use the SemEval-2017 Task 4 Twitter dataset. The GloVe embeddings (~1.5GB) will be downloaded automatically on first run.

## Data source

https://alt.qcri.org/semeval2017/task4/index.php?id=data-and-tools