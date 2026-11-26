---
pubDatetime: 2024-04-16T00:00:00Z
title: "Continual Learning: Which metrics are important?"
slug: continual-learning-metrics
description: "Ranking the most valuable metrics for monitoring deployed ML models, from user feedback to distribution drift."
tags:
  - MLOps
  - distribution drift
  - continual learning
---

One of the hardest questions in the lifecycle of a deployed machine learning model is: "when do I need to retrain my model?". The answer comes from monitoring and observability: detecting when the model is no longer performing well on production data. The key questions to ask yourself as an ML Engineer are:

- Which metrics actually matter?
- How can we use these metrics to debug issues with the model?
- As we get confidence in which metrics are useful, how can we automate this?

There are no settled standards for model monitoring yet. The main principles to follow are:

1. Alert on metrics that indicate the system is breaking.
2. Calculate other metrics too, but use them for observability and debugging.

## Which metrics to monitor?

Choosing the right metrics is the most important decision in setting up useful monitoring. Here are the types of metrics, ranked from most to least valuable:

1. Outcomes and feedback from users
2. Model performance metrics
3. Proxy metrics
4. Data quality
5. Distribution drift
6. System metrics

### Outcomes and Feedback from users

The most valuable signal is outcome data and feedback from your users. There is no universal approach to collecting it, because it depends on the product. It is essentially a product design problem: building the product so that feedback is captured naturally as part of the user experience.

### Model Performance Metrics

Next are model performance metrics such as accuracy. They are less informative than user feedback because the metric you optimise is rarely the outcome you care about. Many practitioners have found that improving a model metric leaves product outcomes unchanged, or even makes them worse. That is no excuse to ignore them, though. Measuring performance in production requires labels, so budget for labelling production data regularly, for example through an on-call rotation or periodic labelling sessions. This shows you how the model's performance changes over time.

### Proxy Metrics

Proxy metrics are signals that correlate with poor model performance but do not need labels. They are usually domain-specific. For a language model generating text, the rate of repetitive or toxic outputs is a useful proxy. In a recommender system, the proportion of personalised recommendations can be indicative. Known failure modes also work as proxies: if your model repeatedly fails on a particular kind of edge case, a rise in how often that case occurs may signal degraded performance.

There is growing research interest in estimating performance metrics on unlabelled production data. Approaches range from training auxiliary models that predict your primary model's performance, to heuristics and human-in-the-loop techniques.

However, no method can estimate performance reliably under every kind of distribution shift. For instance, if the relationship between inputs and labels changes (concept drift), the inputs alone may look unchanged, so any estimate that sees only unlabelled data can miss the degradation entirely.

### Data Quality

In practice, many production ML failures come from the data and the pipeline around the model rather than the model itself. In this [Google talk at USENIX OpML '20](https://www.usenix.org/conference/opml20/presentation/papasian), which reviewed roughly a decade of outages in one of Google's largest ML pipelines, most outages were not ML-specific: they stemmed from the surrounding, distributed data pipeline. Checking data quality (missing values, schema changes, out-of-range values, stale or delayed data) is cheap and catches many of these failures.

### Distribution Drift

#### Why measure distribution drift?

- Performance measured at validation time only holds for data drawn from the same distribution the model was trained and evaluated on.
- Inputs that are rare or absent in training data can have serious consequences, for example the [series of Tesla Autopilot crashes into emergency vehicles investigated by NHTSA](https://cbsaustin.com/news/spotlight-on-america/responders-at-risk-nhtsa-probes-driver-assistance-systems-after-a-series-of-crashes-involving-teslas-and-emergency-vehicles).

It also helps to be clear about what has shifted. *Covariate drift* is a change in the input distribution P(x). *Label drift* is a change in P(y). *Concept drift* is a change in P(y|x), the relationship the model has learned. Input drift does not always hurt performance, and concept drift can hurt performance without any visible change in the inputs.

#### Types of Distribution Drift:

- *Instantaneous*: occurs with a move to a new domain, pipeline bugs, or major external shocks (a pandemic, a war).
- *Gradual*: reflects evolving user preferences or new concepts over time (e.g. [mayonnaise overtaking ketchup as Britain's most popular condiment](https://www.joe.co.uk/food/mayonnaise-has-overtaken-ketchup-as-britains-most-popular-condiment-141342); not an ML example, but it shows how preferences shift in a population).
- *Periodic*: arises from seasonal patterns, or from users in different time zones and regions behaving differently (e.g. ice cream sales are higher in summer than in winter).
- *Temporary*: caused by malicious attacks, a new type of user, or unintended usage patterns (e.g. the Do Anything Now (DAN) jailbreak prompts used against ChatGPT).

#### How to measure it:

- Select a "good" reference window of data, typically validation data or a period of production data where the model performed well.
- Choose a recent window of production data and compare the two distributions using a distance metric.

#### Metrics to use:

- For one-dimensional data, estimate the distribution of the feature (e.g. a histogram) in each window and compare them with a distance such as the L-infinity distance, the L1 distance (equivalent to twice the total variation distance), or the [earth mover's (Wasserstein-1) distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html). These give an interpretable magnitude of shift that you can threshold.
- Be careful with hypothesis tests such as the two-sample Kolmogorov-Smirnov (KS) or chi-squared tests. Their p-values measure how confident you can be that the distributions differ at all, not how large the difference is. With lots of data, tiny and harmless shifts produce very small p-values, so it is hard to pick a threshold that corresponds to degraded model performance. The KS statistic itself (the maximum gap between the two empirical CDFs) is still a usable distance. KL divergence is not a test and returns no p-value, but it has its own problems: it is asymmetric, unbounded, and infinite when the new data has mass where the reference has none, so it needs binning and smoothing to be stable. Further details can be found in [this Gantry post](https://web.archive.org/web/20241010111802/https://www.gantry.io/blog/youre-probably-monitoring-your-models-wrong).
- For high-dimensional data, consider multivariate methods like [maximum mean discrepancy](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html) (powerful, but hard to interpret), or project the data to lower dimensions and apply the one-dimensional metrics above. Projections can be analytical (e.g. mean pixel value, sentence length, or any other function of the input), random (e.g. random linear projections), or learned (e.g. an autoencoder's latent space, PCA, or a model's own embeddings).

#### Challenges:
- Models can be robust to some degree of distribution drift.
- In practice, measuring drift requires careful choices of window size, data retention, metric, and projections.

### System Metrics

These tell you nothing about the quality of the model's predictions, but they can tell you when something is going wrong. Monitor latency, throughput and error rates, and track CPU utilisation and GPU memory usage to make sure you are using the resources you pay for.
