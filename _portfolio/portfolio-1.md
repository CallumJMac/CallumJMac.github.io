---
title: "Prompt Intent Detection"
excerpt: "A scalable system to detect malicious prompts to an LLM with a rapidly built GUI. <br/><img src='/images/mal_prompt.jpg' width=500>"
collection: portfolio
---

### Background

This project was completed as a coding challenge for a job interview. However, I thought it was a good opportunity to find another similar open-source dataset and to share the project as others may find it useful!

The brief specified a lightweight, scalable solution that was computationally efficient. Further, the recruiter was lookng for well documented code and justified decision making in the design of the system.

The code to the project can be found [here](https://github.com/CallumJMac/prompt_classification).

### Methodology

Given the labelled text data in the provided dataset, this challenge can be solved as a text
classification problem. The design requirements specify a lightweight and scalable solution;
therefore, classic machine learning methods such as Logistic Regression, Support Vector
Machine, Random Forest, Gradient Boosting, and K-Nearest Neighbour were chosen over
complex LLMs like BERT, as transformer models have quadratic mathematical complexity in
terms of the number of tokens in the input sequence.
Before inputting the text data into the model, the data needs to be preprocessed and
transformed into a feature vector. The preprocessing steps undertaken were:
- Tokenization: breaks text into individual words or tokens.
- Stopword and punctuation removal: remove common words and punctuation marks
to reduce noise and focus on more semantically meaningful content in text data.
- Lowercasing: Standardises text by converting all characters to lowercase, treating
words with different cases as semantically equivalent.
- Stemming: reduces words to their base or root form to standardise text, reducing
vocabulary size and improving generalisation.
- Count Vectorizer: converting text data into a sparse numerical feature vector based
on word counts. Sparse feature vectors were used as the design specification states
the system should be scalable and prioritise fast inference for a lightweight solution.
Word embeddings are dense representations that capture semantic relationships
between words in a continuous vector space. While they are more expressive and
capture more nuanced information, they typically require more memory to store due
to their dense nature and higher dimensionality.


### Challenges Encountered:
- The trade-off between speed and performance: Word embeddings and Transformer
models may capture a more semantically meaningful relationship but have higher
inference costs.

### Improvements:
- Modularization: smaller modules to improve readability and maintainability.
- Error Handling: Implement error handling mechanisms to handle exceptions.
- Logging: log important information and debug messages during model training.
- Parameter Tuning: optimise for performance on representative data.
- Add user feedback to create user-driven curation loops for continual learning.
- Unittest or PyTests for test-driven development.
- Cross-validation: assess the models' generalisation and robustness.
- Class balancing: This should be investigated to ensure the model is not biased.
- Experiment with model pruning, quantization, or model distillation to reduce model
size and improve inference speed.
- Deploy the system on a scalable infrastructure, such as cloud-based services, to
handle varying workloads and ensure availability.
- Build an API using Django/Flask/FastAPI. Develop a custom GUI using JavaScript
