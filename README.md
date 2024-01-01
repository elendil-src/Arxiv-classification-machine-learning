# Arxiv-classification-machine-learning

## Overview
Notebook to generate model for classifying set of Arxiv Computer Science artices, and Flask-based REST API to serve predictions.

## Instructions
- Python 3.8 at a minimum is required.

- Unzip ```arxiv-dataset/arxiv_data.csv.7z```

- Setup Python packages; for reference ```pip-list.txt``` contains a superset of the packages installed locally.

- Run Jupiter notebook from top level folder  to generate models and text vectorizer, which are saved to this folder.

- Run api.py which setups API

- Execute sample REST reqests in ```classifification-requests.http``` using a VScode http client extension. Response contains results of the classification request for each classification category.



