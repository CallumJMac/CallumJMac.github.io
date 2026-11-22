---
pubDatetime: 2024-05-21T00:00:00Z
modDatetime: 2026-11-22T00:00:00Z
title: "Advanced Retrieval for Retrieval-Augmented Generation"
slug: advanced-retrieval-rag
description: "Query expansion, cross-encoder re-ranking, and embedding adaptors for improving RAG retrieval quality."
tags:
  - LLM
  - LLMOps
  - RAG
  - Information Retrieval
---

Please see the link to the GitHub Repo at the end of the post!

## Introduction

Simple vector search often falls short in retrieval. Chunks that are close to the query in a general-purpose embedding space are not necessarily the ones that answer it. This post covers advanced retrieval techniques for Retrieval-Augmented Generation (RAG): the geometric intuition behind embedding spaces, relevant results versus distractors, and ways to improve retrieval with query expansion, re-ranking and user feedback.

<p align="center">
  <img src="/images/4_advanced retrieval/librarian.png" alt="A librarian searching for a book with a really large magnifying glass in an enormous library" />
  <br />
  <em>Figure 1: AI Generated Image with the prompt "A librarian searching for a book with a really large magnifying glass in an enormous library"</em>
</p>


## Intuition Behind Embedding Spaces
Embedding spaces are abstract and high-dimensional. The model used here, all-MiniLM-L6-v2 (the default in Chroma's `SentenceTransformerEmbeddingFunction`), produces 384-dimensional vectors; BERT-base produces 768, and OpenAI's text-embedding-3-large produces 3072. To build intuition, we can project these vectors down to two dimensions with UMAP (Uniform Manifold Approximation and Projection).
 
### Why UMAP?
PCA (Principal Component Analysis) is a linear projection onto the directions of greatest variance, so it preserves global structure but often blurs local neighbourhoods. t-SNE and UMAP are non-linear methods that focus on preserving local neighbourhood structure. Compared with t-SNE, UMAP is faster and, importantly here, can be fitted once and then used to `transform` new points (such as queries) into the same 2D space. Neither method preserves distances faithfully: distances between clusters, and cluster sizes, in the 2D plot should not be over-interpreted.

### Projecting Embeddings with UMAP
In the code below, UMAP is fitted once on all the document embeddings, with fixed seeds for reproducibility. Every point, documents and later queries, is then projected with the same fitted `umap_transform`, so they all land in one consistent 2D space. Following the course, `project_embeddings` calls `transform` one embedding at a time. That is not required for correctness: `transform` also accepts a batch, but because UMAP's transform is stochastic, projecting one at a time with a fixed `transform_seed` gives each point the same projection regardless of what else is passed in with it. It is slower, so batching is fine if you do not need that.

```python
from helper_utils import load_chroma, word_wrap
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import umap
import numpy as np
from tqdm import tqdm


embedding_function = SentenceTransformerEmbeddingFunction()
chroma_collection = load_chroma(filename='microsoft_annual_report_2022.pdf', collection_name='microsoft_annual_report_2022', embedding_function=embedding_function)
chroma_collection.count()

embeddings = chroma_collection.get(include=['embeddings'])['embeddings']
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings) # random_state and transform_seed fixed for reproducible results

def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2)) #2 dimensions
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings   
```

We can now project the embeddings from 384 dimensions to 2 and plot them:

```python
# Project the document chunk embeddings
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

import matplotlib.pyplot as plt

plt.figure()
plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10)
plt.gca().set_aspect('equal', 'datalim')
plt.title('Projected Embeddings')
plt.axis('off')
```

<p align="center">
  <img src="/images/4_advanced retrieval/1.png" alt="Scatter plot of the UMAP projection of all document chunk embeddings" />
  <br />
  <em>Figure 2: UMAP 2D projection of the 384-dimensional chunk embeddings</em>
</p>

The 2D plot can look odd because a high-dimensional space has been squashed into two dimensions, but it still gives useful geometric intuition for what a retrieval system is doing.

## Relevancy and Distraction in Retrieval

Each chunk an information retrieval system returns is either relevant or a distraction. Figure 3 shows an in-scope query and the chunks retrieved for it by vector similarity search. The query embedding is the red 'X' and the retrieved chunks are the green 'O's.

<p align="center">
  <img src="/images/4_advanced retrieval/2.png" alt="UMAP projection with an in-scope query and its retrieved chunks highlighted" />
  <br />
  <em>Figure 3: UMAP Projection of an In scope query</em>
</p>

The retrieved points are not always the query's nearest neighbours in the 2D plot. That is mostly projection distortion: retrieval happens in the original 384-dimensional space, and UMAP cannot preserve every neighbourhood. Separately, even true nearest neighbours are not always relevant, because a general-purpose embedding model knows nothing about your task or queries.

### Understanding Distractors
Distractors are retrieved chunks that are irrelevant to the query. They are hard to diagnose and debug, and they can mislead the LLM that consumes them.

Imagine the document embeddings as a cloud. Vector search always returns the top k nearest neighbours, however far away they are. A query that lands outside the cloud has no close neighbours, so its top k can be scattered across different parts of the cloud (Figure 4). This is why an off-topic query still returns a full set of results, all of them distractors.

<p align="center">
  <img src="/images/4_advanced retrieval/3.png" alt="UMAP projection with an out-of-scope query and its scattered retrieved chunks" />
  <br />
  <em>Figure 4: UMAP 2D Projection of an Out of Scope query</em>
</p>

## Improving Retrieval

Query expansion is a long-standing information retrieval technique. Large language models (LLMs) make it easy to rewrite or enrich the query before it is sent to the vector database. This section covers two variants: expansion with a generated answer, and expansion with multiple queries.

### Query Expansion with Example Answers

Here an LLM generates a hypothetical answer to the query, and that answer is used to help retrieval. The intuition is that a plausible answer looks more like the relevant document chunks than a short question does. [Jagerman et al. (2023)](https://arxiv.org/abs/2305.03653) study this kind of LLM-based query expansion; a closely related method that embeds the generated answer on its own is [HyDE (Gao et al., 2022)](https://arxiv.org/abs/2212.10496).

<p align="center">
  <img src="/images/4_advanced retrieval/Query Expansion with Generated Answers.png" alt="Diagram of query expansion with a generated answer" />
  <br />
  <em>Figure 5: Query Expansion using Example Answers</em>
</p>

#### How It Works: 
1. **Generate a Response**: Take the original query and prompt the LLM to generate a hypothetical answer. For example, you might use a prompt like: "You are an expert helpful financial research assistant. Provide an example answer to the given question, that might be found in a document like an annual report."
2. **Concatenate and Query**: Concatenate the original query and the generated answer, and use this expanded query to search the vector database as usual.

Here is the code:

```python
def augment_query_generated(query, model="gpt-3.5-turbo"):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Provide an example answer to the given question, that might be found in a document like an annual report. "
        },
        {"role": "user", "content": query}
    ] 

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

original_query = "Was there significant turnover in the executive team?"
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"
print(word_wrap(joint_query))
```
Outputs the following expanded query:
`Was there significant turnover in the executive team? Over the past
fiscal year, there have been no significant turnovers within the
executive team. The leadership continuity has provided stability and
consistency in driving our strategic initiatives forward and achieving
our corporate objectives.`

<p align="center">
  <img src="/images/4_advanced retrieval/4.png" alt="UMAP projection showing the original query and the query expanded with a generated answer" />
  <br />
  <em>Figure 6: UMAP 2D Projection of Query Expansion with Example/Generated Answers</em>
</p>

In Figure 6 the red X is the original query and the orange X is the expanded query. Adding an answer written in the style of an annual report moves the query embedding substantially, towards a different region of the document cloud. That can retrieve more relevant chunks, but note the risk: the generated answer is hallucinated (here it asserts there was no turnover), so it can also pull retrieval towards content that merely sounds like the made-up answer.

### Query Expansion with Multiple Queries
The second variant uses an LLM to generate several related queries. This diversifies the search and can improve recall for broad questions.

<p align="center">
  <img src="/images/4_advanced retrieval/Query Expansion with Multiple Queries.png" alt="Diagram of query expansion with multiple queries" />
  <br />
  <em>Figure 7:  Query Expansion with Multiple Queries</em>
</p>

#### How It Works:
1. **Generate Additional Queries**: Prompt the LLM to suggest additional related questions. For instance: "You are a helpful expert financial research assistant. Your users are asking questions about an annual report. Suggest up to five additional related questions to help them find the information they need, for the provided question. Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic. Make sure they are complete questions, and that they are related to the original question. Output one question per line. Do not number the questions."
2. **Retrieve Results**: Use the original query and the additional queries to retrieve results from the vector database.
3. **Aggregate and Process**: Pool and deduplicate the retrieved chunks, then pass them to the LLM (ideally after re-ranking, see below).

```python
def augment_multiple_query(query, model="gpt-3.5-turbo"):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about an annual report. "
            "Suggest up to five additional related questions to help them find the information they need, for the provided question. "
            "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
            "Make sure they are complete questions, and that they are related to the original question."
            "Output one question per line. Do not number the questions."
        },
        {"role": "user", "content": query}
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content
```


In Figure 8 the red X is the original query and the orange X's are the generated queries, which reach different parts of the document cloud and so cover different aspects of the original question.

<p align="center">
  <img src="/images/4_advanced retrieval/5.png" alt="UMAP projection showing the original query and the generated related queries" />
  <br />
  <em>Figure 8: UMAP 2D Projection of Query Expansion with Multiple Queries</em>
</p>

#### Considerations
- **Prompt Engineering**: Results depend heavily on the prompt, which tends to grow long and complex.
- **Redundancy**: Different queries often retrieve the same chunks, so deduplicate them.
- **Precision**: More queries mean more retrieved chunks, many of them irrelevant. We need a way to score each chunk against the original query, which is where re-ranking comes in.

## Cross-Encoder Re-Ranking

Cross-encoder re-ranking scores each retrieved chunk for relevance to the query and reorders the chunks so the most relevant come first.

#### How it Works
1. **Over-retrieve**: Query the vector database for more chunks than you need, since irrelevant ones will be discarded later.
2. **Re-rank**: Score each chunk against the original query with a cross-encoder and sort by score.
3. **Select**: Pass only the top-ranked chunks to the LLM as context.

### Obtain a "Long Tail" of results

For example, retrieve 10 or more chunks rather than the typical 3 to 5. This gives a "long tail" of possibly useful results, which raises recall. The cross-encoder then identifies which chunks are actually most relevant to the query, rather than just nearest neighbours in embedding space.

### Cross-Encoders

A bi-encoder (our embedding model) encodes the query and each document separately into vectors, and relevance is the similarity between those vectors. Document vectors can be precomputed and indexed, so this scales to millions of chunks. A cross-encoder takes the query and a document together as one input, so attention operates across both texts, and outputs a single relevance score. This is usually more accurate, but nothing can be precomputed: every query and document pair needs its own forward pass. That is why cross-encoders are used to re-rank a small candidate set from a bi-encoder, not to search the whole corpus.

<p align="center">
  <img src="/images/4_advanced retrieval/Cross-encoders.png" alt="Diagram comparing a bi-encoder and a cross-encoder" />
  <br />
  <em>Figure 9: Bi-encoder vs. cross-encoder</em>
</p>

### Re-ranking with Query Expansion

Re-ranking pairs naturally with query expansion: expand the query, pool and deduplicate the retrieved chunks, then re-rank them against the original query. For clarity, the snippet below starts with a single query and 10 retrieved chunks; with multiple queries you would re-rank the pooled results in the same way.

```python
query = "What has been the investment in research and development?"
results = chroma_collection.query(query_texts=query, n_results=10, include=['documents', 'embeddings'])

retrieved_documents = results['documents'][0]

for document in results['documents'][0]:
    print(word_wrap(document))
    print('')
```

Then score each (query, chunk) pair with a cross-encoder, sort by score and discard the low scorers. The model below is a MiniLM cross-encoder trained on the MS MARCO passage ranking dataset.

```python
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pairs = [[query, doc] for doc in retrieved_documents]
scores = cross_encoder.predict(pairs)

new_ordering = []
for o in np.argsort(scores)[::-1]:
    new_ordering.append(o+1)
```
The table shows, for each chunk in the original vector search order, its cross-encoder score and its new rank after sorting by that score. (`new_ordering` in the code lists the original positions in re-ranked order: 2, 1, 3, 7, 6, 9, 10, 5, 4, 8.)

| Original Rank | Cross-Encoder Score | New Rank |
|---------------|---------------------|----------|
| 1             | 0.98693466          | 2        |
| 2             | 2.644579            | 1        |
| 3             | -0.26802942         | 3        |
| 4             | -10.73159           | 9        |
| 5             | -7.7066045          | 8        |
| 6             | -5.6469955          | 5        |
| 7             | -4.297035           | 4        |
| 8             | -10.933233          | 10       |
| 9             | -7.0384283          | 6        |
| 10            | -7.3246956          | 7        |

These scores are raw logits, not probabilities, so only their relative order is meaningful out of the box. To decide what to discard you can keep the top K chunks, or keep chunks above a score threshold. A threshold of 0 (a sigmoid probability of 0.5) is a common heuristic for this model, but it is not calibrated for your data, so tune it on labelled examples. Here only chunks 1 and 2 clear it, and chunk 3 is borderline.

Combining query expansion (for recall) with cross-encoder re-ranking (for precision) gives the LLM a shorter, more relevant context.

## Embedding Adaptors

Embedding adaptors use feedback on the relevance of retrieved results to improve retrieval. An embedding adaptor is a learned transformation, here a linear map (a matrix), applied to the query embedding after it leaves the embedding model and before the similarity search, as shown below. The document embeddings and the embedding model are left unchanged, so nothing needs re-indexing.

<p align="center">
  <img src="/images/4_advanced retrieval/Embedding Adaptor.png" alt="Pipeline diagram with an embedding adaptor applied to the query embedding before vector search" />
  <br />
  <em>Figure 10: Embedding Adaptor</em>
</p>

### Preparing the dataset

The adaptor is trained on relevance feedback so that queries move towards the chunks that are relevant for your use case. The process:
1. **Data Collection**: Gather a set of queries for your system. You could generate these with an LLM to get started.
2. **Retrieve Initial Results**: Use your vector database to fetch results for each query.
3. **Evaluate Results**: Label each (query, result) pair thumbs up or thumbs down (+1 or -1). In the course an LLM generates these labels; in production they would come from users. Synthetic labels inherit the LLM's mistakes.
    - Relevant results are rated as +1.
    - Irrelevant results are rated as -1.
4. **Prepare Data Arrays**:
    - **Adapter Query Embeddings**: The embedding of the query for each pair.
    - **Adapter Document Embeddings**: The embedding of the retrieved chunk for each pair.
    - **Adapter Labels**: The +1 or -1 rating for each pair.
5. **Build the Dataset**: Convert the arrays into PyTorch tensors and wrap them in a `TensorDataset`.


### Building the Embedding Adaptor Model

The model is a single square matrix of size d by d, where d is the embedding dimension (384 here). It is initialised with random values and learned in a training loop.

- **Inputs**: Query embedding, document embedding, and adaptor matrix.
- **Computation**:
    1. Multiply the query embedding by the adaptor matrix.
    2. Compute the cosine similarity between the adapted query embedding and the document embedding.

In PyTorch:

```python
def embedding_adaptor_model(query_embedding, document_embedding, adaptor_matrix):
    updated_query_embedding = torch.matmul(adaptor_matrix, query_embedding)
    return torch.cosine_similarity(updated_query_embedding, document_embedding, dim=0)

# Initialize the adaptor matrix
mat_size = len(adapter_query_embeddings[0])
adapter_matrix = torch.randn(mat_size, mat_size, requires_grad=True)
```


### Loss Function and Training

- **Loss Function**: Mean squared error between the predicted cosine similarity and the label. Cosine similarity lies in [-1, 1], so the +1 and -1 labels act as targets: relevant pairs are pushed towards similarity 1 and irrelevant pairs towards -1.
- **Training Efficiency**: The model is a single linear layer without bias, so training is cheap.

The loss in PyTorch:

```python
def mse_loss(query_embedding, document_embedding, adaptor_matrix, label):
    return torch.nn.MSELoss()(embedding_adaptor_model(query_embedding, document_embedding, adaptor_matrix), label)
```

A simple training loop (per-example gradient descent, following the course):

```python
min_loss = float('inf')
best_matrix = None
lr = 0.01

for epoch in tqdm(range(100)):
    for query_embedding, document_embedding, label in dataset:
        loss = mse_loss(query_embedding, document_embedding, adapter_matrix, label)

        if loss < min_loss:
            min_loss = loss
            best_matrix = adapter_matrix.clone().detach().numpy()

        loss.backward()
        with torch.no_grad():
            adapter_matrix -= lr * adapter_matrix.grad
            adapter_matrix.grad.zero_()
```

### Impact of the Embedding Adaptor

The adaptor reshapes the query space, stretching directions that help separate relevant from irrelevant chunks and shrinking those that do not. A full matrix can also rotate and mix dimensions, so it is more than a per-dimension reweighting.

One rough way to visualise it, from the course: multiply a vector of all ones by the trained matrix (Figure 11). The output is the sum of each row of the matrix, which hints at which output dimensions are amplified or suppressed. It is a heuristic view, not a complete description of the transformation.

<p align="center">
  <img src="/images/4_advanced retrieval/6.png" alt="Bar chart of a vector of ones transformed by the trained adaptor matrix" />
  <br />
  <em>Figure 11: Embedding Adaptor Matrix Visualised</em>
</p>

In the UMAP projection below, the original query embeddings are scattered and the adapted ones are more concentrated, closer to the region of the documents they were trained to retrieve. That is suggestive, but it is not proof of better retrieval: a 2D projection can distort distances, and the plot uses the same queries the adaptor was trained on. To show an improvement, measure retrieval metrics such as recall@k or nDCG on held-out queries with and without the adaptor.

<p align="center">
  <img src="/images/4_advanced retrieval/embedadapt_scatter.png" alt="UMAP projection of original and adapted query embeddings" />
  <br />
  <em>Figure 12: UMAP projection of original and adapted query embeddings</em>
</p>

## Conclusion

In this blog post we have covered the following:
- How to compute UMAP projections to visualise your vector database in 2 dimensions
- Query expansion using generated answers
- Query expansion with multiple queries
- Cross-encoder re-ranking
- Embedding Adaptors

This has been a summary of a [course](https://learn.deeplearning.ai/courses/advanced-retrieval-for-ai/lesson/1/introduction) I took and I have pushed the code to my [GitHub](https://github.com/CallumJMac/lessons/tree/main/6.%20Advanced%20Retrieval) if you would like to explore it further!
