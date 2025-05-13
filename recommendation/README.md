# Recommendation systems

## Use cases

1. Popularity-Based Recommendations

   - What It Is : Recommend items based on overall popularity (e.g., most clicked, highest ratings).
   - When to Use : Cold-start scenarios (no user data), or as a baseline.
   - Example : just a simple sort on most popular

1. Collaborative Filtering (CF) or Content-based filtering on tags

   - What It Is : Recommend items based on user-item interactions (e.g., "Users who liked X also liked Y").
   - When to Use : When you have user interaction data (e.g., clicks, ratings).
   - Example : `from sklearn.metrics.pairwise import cosine_similarity` or `from transformers import BertTokenizer, TFBertModel` 

1. Term Frequency (TF):  

   - What It Is : Recommend items similar to those a user has interacted with, using item metadata (e.g., tags, descriptions).
   - When to Use : When item metadata is available (e.g., movie genres, product descriptions).
   - Example : TF-IDF and cosine similarity `from sklearn.feature_extraction.text import TfidfVectorizer` or `const natural = require('natural'); const TfIdf = natural.TfIdf;` TF-IDF (Term Frequency-Inverse Document Frequency)  is a numerical statistic used in text mining and information retrieval to reflect the importance of a term (word) in a document relative to a collection of documents (corpus).  or # Load pre-trained BERT model for sentence embeddings `from model = SentenceTransformer('all-MiniLM-L6-v2')`

1. Matrix Factorization

   - What It Is : Decompose the user-item matrix into latent factors (e.g., user preferences, item attributes).
   - When to Use : For scalable collaborative filtering.
   - Example : `from surprise import Dataset, Reader, SVD` or Neural Collaborative Filtering using pyTorch `import torch.nn as nn`


## Libraries

### Typescript

- Tensorflow : `import * as tf from '@tensorflow/tfjs-node';`
- KNN (Graph) : `import { knn } from 'ml-knn';`
- Maths.js (Matrix) : `import { matrix, multiply, transpose } from 'mathjs';`
- Transformerjs 
- natural :  `npm i natural`

### Python

- sklearn : `from sklearn.metrics.pairwise import cosine_similarity`, `from sklearn.feature_extraction.text import TfidfVectorizer`
- transformer : `from transformers import BertTokenizer, TFBertModel`
- surprise : `from surprise import Dataset, Reader, SVD`
- PyTorch Neuronal networks : `import torch.nn as nn`


## When to use Transformers ?

- **Inference/Prediction** needed : **Session-based** recommendations (e.g., "What will the user click next?").
- Rich **TEXTUAL** Context : Items with textual descriptions (use BERT embeddings). 
- Complex Patterns : **Replace traditional latent factors** with Transformer attention mechanisms.

