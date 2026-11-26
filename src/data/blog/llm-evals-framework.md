---
pubDatetime: 2024-05-07T00:00:00Z
title: "LLM Evals: A General Framework for Custom Evaluations"
slug: llm-evals-framework
description: "A general framework for building rule-based and model-graded evaluations for LLM-based applications."
tags:
  - LLM
  - safety
  - LLMOps
  - CI
  - Red-teaming
---

<p align="center">
  <img src="/images/3_llm_evals/1.jpg" alt="LLM EVALS HOW TO BUILD TESTING LARGE LANGUAGE MODELS" />
  <br />
  <em>Figure 1: AI Generated Image with the prompt "Small person testing an enormous and complex machine in the style of a marvel comic"</em>
</p>


## Introduction

In a recent blog, I introduced the concept of testing Large Language Models (LLMs). Testing LLMs is a complex topic that deserves a closer look. In this blog I propose a general framework that serves as a minimum recommendation for testing applications that use LLMs, such as conversational assistants, retrieval augmented generation (RAG) systems and agents.

### Traditional software vs. LLMs

In software, we write unit tests that are small and test isolated bits of logic that can easily be tested independently and quickly. In machine learning, models are essentially functions that map input → output. However, models can be large, opaque objects with failure modes that make testing them more challenging and nuanced.

Also, software tests usually assess deterministic logic, i.e. the same input gives the same output. Autoregressive LLMs like the GPT family generate text one token at a time from a probability distribution over the next token. With sampling (temperature above zero) the same input can produce many different outputs. Greedy decoding (temperature 0) is close to deterministic, but hosted APIs still do not guarantee identical outputs across calls, and small changes to the prompt can change the answer. On top of that, open-ended tasks often have many acceptable answers. This makes building tests and evaluations for LLMs more challenging.

To demonstrate that an LLM may have many different correct or incorrect solutions, consider the following example:

**Question: "Is London the best city in the world?"**

Answer 1: "Yes"

Answer 2: "Many people think London is the best city in the world due to its cultural diversity, with over 270 nationalities and 300 languages spoken, it is seen as the capital of the world. Additionally, the London skyline is a combination of listed heritage structures that are rich in history and globally recognised combined with iconic structures like the Shard and the Eiffel Tower."

Answer 3: "Determining whether London is the "best" city in the world is subjective and depends on individual preferences, priorities, and criteria for what makes a city great. London certainly has many attributes that make it a highly desirable city for many people, including its rich history, cultural diversity, world-class museums and galleries, vibrant arts and entertainment scene, and economic opportunities. However, whether it is the "best" city overall is open to debate and varies from person to person."

Out of the three generated answers, which one is the correct one that you want your application to generate? 

The first two answers present a subjective opinion as fact, stating that London is the best city in the world. For sensitive use cases, this kind of bias can be extremely undesirable. Answer 2 also states that the Eiffel Tower is in London, which is factually incorrect. This is an example of another vulnerability in LLMs: hallucination, where the model makes statements that sound compelling but are false. One reason is the training objective: LLMs are trained to predict a likely next token, which rewards fluent, plausible text rather than verified facts. A loose analogy is the next word suggestion on your phone keyboard, at a vastly larger scale. Because each token is conditioned on the tokens before it, an early mistake can also be carried forward into the rest of the response. Gaps or errors in the training data and the decoding strategy contribute too.

Other key considerations for LLMs are:
- Effectiveness (accuracy and utility)
- Performance (speed and functionality)
- Quality (user experience, readability and tone)

See below a summary table to compare the features of traditional software and LLM-based applications that are relevant for testing and evaluating them:


|               | Traditional Software             | LLM-based applications                           |
|---------------|----------------------------------|--------------------------------------------------|
| **Behaviour** | Predefined rules                 | Probability + Prediction                          |
| **Output**    | Deterministic   (same input → same output)                 | Stochastic      (same input → many possible outputs)                          |
| **Testing**   | 1 input, 1 correct output       | 1 input, many correct (and incorrect) outputs    |
| **Criteria**  | Evaluate as "right" or "wrong"  | Evaluate on: accuracy, quality, consistency, bias, toxicity, and more |




### Evaluating LLMs vs Evaluating LLM systems

While the primary focus of this blog post is on evaluating LLM systems, it's important to differentiate between assessing a base LLM and evaluating a system that uses LLMs for a specific use case. State-of-the-art (SOTA) LLMs perform a variety of tasks well, including chat, Named Entity Recognition (NER), text generation, summarisation, question answering, sentiment analysis and translation. Base or foundation models are typically evaluated against standardised public benchmarks. Some key examples include:

- **MMLU (Massive Multitask Language Understanding)**: multiple choice questions across 57 subjects, from elementary mathematics to law and medicine, testing broad knowledge and problem solving.

- **HellaSwag**: a commonsense reasoning benchmark where the model picks the most plausible continuation of an everyday scenario from four options, with wrong endings adversarially filtered to be hard for models.

- **GLUE (General Language Understanding Evaluation)** and **SuperGLUE**: collections of sentence and sentence pair understanding tasks (e.g. entailment, sentiment, paraphrase). SuperGLUE is the harder successor; both are now largely saturated by modern LLMs.

- **TruthfulQA**: questions designed so that common misconceptions and falsehoods are tempting answers, measuring whether a model avoids repeating them.

Strong benchmark scores do not guarantee that a model works "out of the box" for a specific use case, which may require fine-tuning on a proprietary dataset or adding retrieval. Evaluating a fine-tuned or RAG-based system usually means comparing its outputs against a ground truth dataset, if one is available. The responsibility extends beyond the model itself: it is the engineer's job to ensure the whole application produces the desired outputs, through appropriate prompt templates, effective retrieval pipelines, the choice of model (and fine-tuning, if required) and more. Selecting the right components and evaluating the full system remains a nuanced challenge.

We are going to consider evaluations not for building base LLMs, but for niche and specific use cases. We are going to assume that we are using a model and trying to build an application such as a chatbot for a bank, or a 'chat to your data' application. Further, we want our tests to be automatic and scalable to allow us to iteratively improve our application throughout the Machine Learning Operations (MLOps) lifecycle! How do we build our own metrics?

### Automating Evaluations (Evals)

*What* should we evaluate?

1. **Context adherence** (or groundedness): is the LLM response supported by the provided context, and does it follow the given guidelines?

2. **Context relevance**: for systems that use RAG, is the retrieved context relevant to the original query or prompt?

3. **Correctness**: (or accuracy) does the LLM output align with ground truth or the expected results?

4. **Bias and toxicity**: harmful behaviours that must be mitigated. Bias is prejudice towards or against certain groups. Toxicity is harmful, abusive or offensive language, whether explicit or implicit.

*When* should we evaluate?

1. **After every change**: as in software, test after every change, whether that is a bug fix, a feature update, or a change to the data, model or prompt template.

2. **Pre-deployment**: if testing after every change becomes too costly or slow, run a more comprehensive suite at key points instead, such as when merging a branch to main, before releasing to production, or at the end of each sprint.

3. **Post-deployment**: the form of ongoing evaluation depends on the use case and business requirements, but it is essential for detecting drift and improving your system over time. Collecting user feedback and running A/B tests are good ways to iteratively improve the user experience.

### Rule Based Evaluations

One of the simplest things we may want to ensure when testing our generated responses is:

- Are there certain pieces of text that our system is expected to generate?

- Are there certain pieces of text that the system can not generate?

The risk profile of the project will likely dictate that certain words must never be used, and there may be words or phrases we expect our system to generate. The examples below assume a small `assistant_chain` helper that builds a LangChain chain from a system message and prompt template. Here is how we can test for expected words:

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def assistant_chain(system_message, human_template, llm, output_parser):
    """Build a simple prompt | llm | parser chain."""
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_template),
    ])
    return chat_prompt | llm | output_parser


def eval_expected_words(
    system_message,
    question,
    expected_words,
    human_template="{question}",
    # Language model used by the assistant
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    # Output parser to parse assistant's response
    output_parser=StrOutputParser()
):
    """
    Evaluate if the assistant's response contains the expected words.

    Parameters:
        system_message (str): The system message sent to the assistant.
        question (str): The user's question.
        expected_words (list): List of words expected in the assistant's response.
        human_template (str, optional): Template for human-like response formatting.
            Defaults to "{question}".
        llm (ChatOpenAI, optional): Language model used by the assistant.
            Defaults to ChatOpenAI(model="gpt-3.5-turbo", temperature=0).
        output_parser (OutputParser, optional): Output parser to parse assistant's response.
            Defaults to StrOutputParser().

    Raises:
        AssertionError: If the expected words are not found in the assistant's response.
    """
    
    # Create an assistant chain with provided parameters
    assistant = assistant_chain(
        system_message,
        human_template,
        llm,
        output_parser
    )
    
    # Invoke the assistant with the user's question
    answer = assistant.invoke({"question": question})
    
    # Print the assistant's response
    print(answer)
    
    # Check if any of the expected words are present in the assistant's response.
    # If not, raise an AssertionError so the test runner reports a failure.
    assert any(word.lower() in answer.lower() for word in expected_words), (
        f"Expected the assistant response to include one of {expected_words}, but it did not"
    )

```

This example is written to evaluate a system built with LangChain. However, the logic is simple and could be applied to rule-based evaluations for testing systems in other frameworks too.

Alternatively, if we ask the application to generate text on something undesirable, we may hope that the application declines to generate text on the undesirable topic. The next example gives a template for testing if the system can refuse certain topics or words.

```python
def evaluate_refusal(
    system_message,
    question,
    decline_response,
    human_template="{question}", 
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    output_parser=StrOutputParser()):
    """
    Evaluate if the assistant's response includes a refusal.

    Parameters:
        system_message (str): The system message sent to the assistant.
        question (str): The user's question.
        decline_response (str): The expected response from the assistant when refusing.
        human_template (str, optional): Template for human-like response formatting.
            Defaults to "{question}".
        llm (ChatOpenAI, optional): Language model used by the assistant.
            Defaults to ChatOpenAI(model="gpt-3.5-turbo", temperature=0).
        output_parser (OutputParser, optional): Output parser to parse assistant's response.
            Defaults to StrOutputParser().

    Raises:
        AssertionError: If the assistant's response does not contain the expected refusal.
    """
    
    # Create an assistant chain with provided parameters
    assistant = assistant_chain(
        system_message,
        human_template,
        llm,
        output_parser
    )
  
    # Invoke the assistant with the user's question
    answer = assistant.invoke({"question": question})
    # Print the assistant's response
    print(answer)
    
    # Check if the expected refusal is present in the assistant's response
    assert decline_response.lower() in answer.lower(), (
        f"Expected the bot to decline with '{decline_response}', got {answer}"
    )
```

These examples can serve as templates for rule-based evals, and the same approach can use regular expressions to test for patterns. For example, you may wish to run a rule-based eval on:
- ranges of dates
- ensuring the system does not generate personal information (bank account numbers, unique identifiers etc.)

To design a rule-based eval, use the following steps:
1. **Define Evaluation Function**: As in the examples provided, define evaluation functions tailored to specific evaluation criteria. These would take inputs like system messages, user questions and expected responses.
2. **Define Logic Check**: This could be exact matches, simple containment checks, or you could use regular expressions to define more complex patterns for expected responses. 
3. **Incorporate Logic Check into Assertions**: Modify the assertion checks in the evaluation function to use the defined logic for your use case, to reflect expected behaviour for your system.
4. **Documentation**: Include docstrings to describe the purpose of each evaluation function, the expected inputs and the evaluation criteria. The documentation helps maintain clarity and facilitates collaboration in your team.
5. **Error handling**: Make failures explicit with clear assertion messages, and do not catch and silence assertion errors, so failures are properly reported and diagnosed by your test runner.
6. **Parameterisation**: Consider using threshold values to allow you to adapt evaluation criteria depending on the risk profile of your use case. 

### Model-Graded Evaluations

How can we better ensure the overall quality of the generated responses in our system? We want to have an evaluation method that is more robust and comprehensive. This is tricky, as what makes a "good" response can be highly subjective. Rule-based evals are useful for checking that certain information is or is not in our output, but they become brittle as the application grows.

One method is to use an LLM to grade the outputs of an LLM app, known as model-graded evals (or LLM-as-a-judge). Yes, that's right, we are using AI to check AI.

In this code snippet, we are implementing a mechanism to evaluate the quality of the responses generated. Specifically, we are looking at the format of the response, although any feature of the generated text could be considered by adapting the `eval_system_prompt` and the `eval_user_message`. The `create_eval_chain` function builds the grading chain that evaluates the main LLM application.

```python
def create_eval_chain(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    output_parser=StrOutputParser()
):
    """
    Creates an evaluation chain to assess the appropriateness of the agent's response.

    Parameters:
        llm (ChatOpenAI, optional): Language model used for evaluation.
            Defaults to ChatOpenAI(model="gpt-3.5-turbo", temperature=0).
        output_parser (OutputParser, optional): Output parser for parsing agent's response.
            Defaults to StrOutputParser().

    Returns:
        Runnable: Evaluation chain; invoke it with {"agent_response": ...}.
    """

    eval_system_prompt = """You are an assistant that evaluates whether or not an assistant is producing valid responses.
    The assistant should be producing output in the format of [CUSTOM EVALUATION DEPENDING ON USE CASE]."""

    # {agent_response} is a template variable, filled in when the chain is invoked.
    # Avoid f-strings here: braces in the response would break the template.
    eval_user_message = """You are evaluating [CUSTOM EVALUATION DEPENDING ON USE CASE].
    Here is the data:
      [BEGIN DATA]
      ************
      [Response]: {agent_response}
      ************
      [END DATA]

    Read the response carefully and determine if it [MEETS REQUIREMENT FOR USE CASE]. Do not evaluate whether the information is correct, only whether the data is in the expected format.

    Output only 'True' if the response is appropriate, or only 'False' if it is not.
    """
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", eval_system_prompt),
        ("human", eval_user_message),
    ])

    return eval_prompt | llm | output_parser
```

The `create_eval_chain` function returns a chain (prompt, grading model, output parser) that assesses a given response. You choose the model and output parser used for evaluation, and pass the generated response when invoking the chain.

We ask the grading model to output 'True' or 'False' so that results are easy to interpret as pass or fail across many tests, which makes the testing regime scalable.

As the LLM outputs a string, we need some post-processing to turn it into a test result. The following function `model_grad_eval_format()` could be reused for processing all our model-graded evals in a given test suite. 

```python
def model_grad_eval_format(generated_output):
    """
    Evaluates the format of the generated output from the agent.

    Parameters:
        generated_output (str): The output generated by the agent.

    Returns:
        bool: True if the response is appropriate, False otherwise.

    Raises:
        ValueError: If the evaluation result is neither "True" nor "False".
    """
    # Create an evaluation chain for assessing the agent's response format
    eval_chain = create_eval_chain()

    # Retrieve the evaluation result from the evaluation chain
    evaluation_results_str = eval_chain.invoke({"agent_response": generated_output})

    # Strip leading/trailing whitespaces and convert to lowercase
    retrieval_test_response = evaluation_results_str.strip().lower()

    # Map the grader's verdict to a boolean
    if retrieval_test_response.startswith("true"):
        return True
    if retrieval_test_response.startswith("false"):
        return False
    raise ValueError(f"Unexpected evaluation result: {evaluation_results_str}")

model_grad_eval_format(generated_output="[UNDESIRED OUTPUT GENERATED BY AGENT]")
```

The `model_grad_eval_format()` function parses the evaluation result into a boolean, or raises a `ValueError` if the grader replies with anything else. Used this way, evals let us systematically assess response quality and identify areas for improvement.

Some further tips on model-based evals:

- **Opt for the most capable model you can afford**: Critiquing outputs often requires strong reasoning. Your deployed system may need a fast, low latency model, but a slower and more powerful LLM may be needed to grade outputs well. (The `gpt-3.5-turbo` defaults above are placeholders.)

- **You could create a problem within a problem**: The grading model can also make mistakes and give you a false sense of confidence. Check its verdicts against a sample of human labels before trusting it. Known LLM-as-a-judge biases include position bias (favouring the first or second answer in pairwise comparisons), verbosity bias (favouring longer answers) and self-preference bias (favouring outputs from the same model family).

- **Create positive and negative evals**: Test the grader with examples that should pass and examples that should fail. A grader that says 'True' to everything is useless, so make sure it can discriminate between the two.


## Conclusion

- Effective evaluation of LLM-based applications requires a multi-faceted approach, encompassing both rule-based and model-graded evals to assess various aspects of system performance.

- You should continually refine and adapt both your system and eval methods iteratively to keep pace with evolving user expectations, system requirements and advancements in LLM technology.

- By embracing a systematic, iterative and observability-based approach to evaluation, stakeholders can gain valuable insights into the strengths and weaknesses of LLM-based applications, driving continuous improvement of LLM applications.

## Further Reading

Please see below a list of relevant resources to this blog post:

[OpenAI Evals Guide](https://cookbook.openai.com/examples/evaluation/getting_started_with_openai_evals)

[Jane Huang's (Microsoft) Blog post](https://medium.com/data-science-at-microsoft/evaluating-llm-systems-metrics-challenges-and-best-practices-664ac25be7e5)

[LlamaIndex Notebook on Evals](https://github.com/run-llama/ai-engineer-workshop/blob/main/notebooks/02_evaluation.ipynb)

[Hamel H.'s blog](https://hamel.dev/blog/posts/evals/#eval-systems-unlock-superpowers-for-free)

[LlamaIndex's Docs](https://docs.llamaindex.ai/en/v0.10.17/optimizing/evaluation/evaluation.html)

[Free course on automated testing for LLMOps with CircleCI](https://learn.deeplearning.ai/courses/automated-testing-llmops/lesson/1/introduction)
