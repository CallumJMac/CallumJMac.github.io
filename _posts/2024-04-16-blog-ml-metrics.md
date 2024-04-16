---
title: 'Continual Learning: Which metrics are important?'
date: 2012-08-14
permalink: /posts/2024/04/16/CL-metrics/
tags:
  - MLOPS
  - distribution drift
  - continual learning
---

One of the most difficult things to identify throughout the lifecycle of a deployed machine learning model is: "when do I need to retrain my model?". The most important thing is to use monitoring and observability to determine when a model is no longer performing well on offline data. The key questions you need to ask yourself as an ML Engineer are:

- Which metrics actually matter?
- How can we use these metrics to debug issues with the model?
- As we get confidence in which metrics are useful, how can we automate this?

The bottom line is, there are no real standards or best practices on model monitoring. The main principles to follow are:
1. Focus on merics that cause the system to break.
2. Calculate other metrics too, but use them for observability and debugging.

## Which metrics to monitor?

Selecting the right meric to monitor is the most important decision to make for useful monitoring. Here is a list of types of metrics ranked from most to least valuable:

1. Outcomes and feedback from users
2. Model performance metrics
3. Proxy metrics
4. Data quality
5. Distribution drift
6. System metrics

### Outcomes and Feedback from users

One of the most crucial aspects to consider is gathering outcome data and feedback from your users. However, there's no universal approach to this, as it heavily relies on the unique characteristics of the product you're developing. It's essentially a product management endeavor, centered around designing your product in a manner that seamlessly integrates user feedback into the overall user experience.

### Model Performance Metrics

Another critical indicator to consider is the performance metrics of your model. These metrics, such as accuracy, provide valuable insights, although they're considered less informative compared to direct user feedback due to potential mismatches in loss functions. It's a common experience among machine learning practitioners to find that enhancing model performance doesn't always translate to improved outcomes, and sometimes even leads to the same or worse results. However, neglecting this aspect is unjustifiable. One effective approach is to allocate resources to label production data regularly, either through establishing an on-call rotation or organizing labeling sessions. These practices offer visibility into the trajectory of your model's performance over time.

### Proxy Metrics

Another valuable approach is examining proxy metrics, which often correlate with poor model performance and are typically domain-specific. For instance, in text generation using a language model, indicators like repetitive or toxic outputs serve as relevant proxies. Similarly, in recommendation systems, the proportion of personalized responses can be indicative. Edge cases also hold significance as proxy metrics. If specific issues persistently arise with your model, an uptick in their occurrence may signal subpar performance.

In academia, there's a growing interest in devising methods to approximate any desired metric on previously unseen data, offering practical utility to these proxy metrics. Various strategies exist, ranging from training auxiliary models to predict the performance of your primary model on offline data, to employing heuristics and incorporating human-in-the-loop techniques.

However, it's worth noting that no single method universally applies to approximate model performance on out-of-distribution data. For instance, changes in label distribution due to evolving input data pose a challenge to accurately assessing model performance using approximate metrics.

### Data Quality

Data quality issues are the most common issue with ML models in practice. In this [Google Conference](https://www.usenix.org/conference/opml20/presentation/papasian), which examined 15 years of different pipeline outages with ML models, the most common cause of outages were data problems.

### Distribution Drift

#### Why measure distribution drift?

- Model performance is only reliable on data from the same distribution it was trained on.
- Distribution drift can significantly impact outcomes, as seen in scenarios like [Tesla's Autopilot crashing into emergency vehicles](https://cbsaustin.com/news/spotlight-on-america/responders-at-risk-nhtsa-probes-driver-assistance-systems-after-a-series-of-crashes-involving-teslas-and-emergency-vehicles).


#### Types of Distribution Drift:
- *Instantaneous*: Occur with domain changes, pipeline bugs, or major external shifts (pandemic, wars, etc.)
- *Gradual*: Reflects evolving user preferences or new concepts over time. (E.g. [Mayonnaise overtaking ketchup as Britain's most popular condiment](https://www.joe.co.uk/food/mayonnaise-has-overtaken-ketchup-as-britains-most-popular-condiment-141342)- Not strictly ML, but demonstrates evolving preferences in the general population).
- *Periodic*: Arises from seasonal or geographic variations in user behaviour. (E.g. seasonal ice cream purchasing - more in summer and less in winter).
- *Temporary drift*: Resulting from malicious attacks, new user behaviour, or unintended usage patterns (e.g. Do Anything Now (DAN) prompt injection attack on ChatGPT).

#### How to measure it:
- Select a "good" data window for reference, often using production or validation data.
- Choose a new production data window and compare distributions using distance metrics.

#### Metrics to use:
- For one dimensional data, compute a probability density of the feature and use the inf-norm, 1-norm, or [earth-mover's distance](https://arxiv.org/abs/1904.05877).
- KL divergence and KS test are less useful as they try to return a p-value for the likelihood the data distributions are not the same. When you have lots of data, very small p-values will be outputted for small shifts. This often makes it difficult to determine a p-value that indicates a degradation in model performance. Further details can be found [here](https://www.gantry.io/blog/youre-probably-monitoring-your-models-wrong).
- For high-dimensional data, explore methods like [maximum mean discrepancy](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html) (these can be challenging to interpret) or projections to reduce the dimensionality of the features. analytical projections (e.g., mean pixel value, length of sentence, or any other function), random projections (e.g., linear), and statistical projections (e.g., autoencoder or other density models, T-SNE).

#### Challenges:
- Models can be robust to some degree of distribution drift.
- Practically, measuring distribution drift requires careful window sizing, data retention, metric selection, and defining projections for lower-dimensional analysis.

### System Metrics

These don't tell you anything about how the model is actually performing, but they can tell you when something is going wrong. Further, it is important that you utilise the resources that you pay for. Therefore, monitor CPU utilization and GPU memory usage etc. 