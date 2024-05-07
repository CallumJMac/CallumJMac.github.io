---
title: 'LLMs Evals: A General Framework for Custom Evaluations'
date: 2024-05-07
permalink: /posts/2024/05/07/LLM_evals/
tags:
  - LLM
  - safety
  - LLMOPs
  - CI
  - Red-teaming
---

*Disclaimer: I am new to blogging. So, if there are any mistakes, please do let me know. All feedback is warmly appreciated.*

<p align="center">
  <img src="/images/3_llm_evals/1.jpg" alt="“LLM EVALS HOW TO BUILD TESTING LARGE LANGUAGE MODELS" />
  <br />
  <em>Figure 1: AI Generated Image with the prompt "Small person testing an enormous and complex machine in the style of a marvel comic"</em>
</p>


## Introduction

In a recent blog, I introduced the concept of testing Large Language Models (LLMs). However, testing Large Language Models (LLMs) is a complex topic that needs to be explored further. There are several considerations about testing Machine Learning models and specifically LLMs that must be considered when developing and deploying your application. In this blog I will propose a general framework that will serve as a minimum recommendation for testing applications that use LLMs, including conversational agents, retrieval augmented generation, and agents etc. 

### Traditional software vs. LLMs

In software, we write unit tests that are small and test isolated bits of logic that can easily be tested independently and quickly. In machine learning, models are essentially functions that map input → output. However, models can be large logic objects that have many additional vulnerabilities and complexities that makes testing them more challenging and nuanced. 

Also, tests in software assess logic that is deterministic, i.e. the same input will result in the same output. In machine learning and particularly autoregressive (predict the next word) LLMs like the GPT family are non-deterministic, i.e. the same input will result in many possible outputs. This makes building tests and evaluations for LLMs more challenging. 

To demonstrate that a LLM may have many different correct or incorrect solutions, consider the following example:

**Question: "Is London the best city in the world?"**

Answer 1: "Yes"

Answer 2: "Many people think London is the best city in the world due to it cultural diversity, with over 270 nationalities and 300 languages spoken, it is seen as the capital of the world. Additionally, the London skyline in a combination of listed heritage structures that are rich in history and globally recognised combined with iconic structures like the Shard and the Eiffel Tower."

Answer 3: "Determining whether London is the "best" city in the world is subjective and depends on individual preferences, priorities, and criteria for what makes a city great. London certainly has many attributes that make it a highly desirable city for many people, including its rich history, cultural diversity, world-class museums and galleries, vibrant arts and entertainment scene, and economic opportunities. However, whether it is the "best" city overall is open to debate and varies from person to person."

Out of the three generated answers, which one is the correct one that you want your application to generate? 

There are some issues associated with the first two generated answers, they state that London is the best city in the world which is considered biased. For sensitive use cases, bias can be extremely undesirable for stakeholders responsible for LLMs. Additionally, answer 2 states that the Eiffel Tower is in London. Obviously, this is nonsense and factually incorrect. This is an example of another vulnerability in LLMs - hallucinations. They have been known to make statements that sound compelling and true, but are actually incorrect. This phenomenon occurs as most LLMs are auto-regressive, they are trained to predict the next word in a sentence. Think of this in the same way that your iPhone suggests the next word when you are typing a text message. Once the model predicts a word that is not contextually appropriate, this can propagate and cause further inappropriate predictions.

Other key considerations for LLMs are:
- Effectiveness (accuracy and utility)
- Performance (speed and functionality)
- Quality (user experience, readability and tone)

See below a summary table to compare the features of traditional software and LLM-based applications that are relevant for testing and evaluating them:


|               | Traditional Software             | LLM-based applications                           |
|---------------|----------------------------------|--------------------------------------------------|
| **Behavior**  | Predefined rules                 | Probability + Prediction                          |
| **Output**    | Deterministic   (same input → same output)                 | Non-deterministic      (Same Input → many possible outputs)                          |
| **Testing**   | 1 input, 1 correct output       | 1 input, many correct (and incorrect) outputs    |
| **Criteria**  | Evaluate as "right" or "wrong"  | Evaluate on: accuracy, quality, consistency, bias, toxicity, and more |




### Evaluating LLMs vs Evaluating LLM systems

While the primary focus of this blog post is on evaluating LLM systems, it's important to differentiate between assessing a base LLM and evaluating a system that uses LLMs for a specific use case. State-of-the-art (SOTA) LLMs perform a variety of tasks extremely well including, chatbot interactions, Named Entity Recognition (NER), text generation, summarization, question-answering, sentiment analysis, translation, and more. Typically, these models undergo evaluation against standardised benchmarks such as GLUE (General Language Understanding Evaluation), SuperGLUE, HellaSwag, TruthfulQA, and MMLU (Massive Multitask Language Understanding). Several benchmark datasets are used to evaluate foundation models or base LLMs. Some key examples include:

- **MMLU (Mean Message Length in Utterance)**: MMLU ((Massive Multitask Language Understanding) evaluates how well the LLM can multitask

- **HellaSwag**: Evaluates how well an LLM can complete a sentence.

- **GLUE**: GLUE (General Language Understanding Evaluation) benchmark provides a standardized set of diverse NLP tasks to evaluate the effectiveness of different language models.

- **TruthfulQA**: Measures truthfulness of model responses.

The immediate applicability of these LLMs "out-of-the-box" may be limited for our specific requirements due to the potential necessity of fine-tuning the LLM using a proprietary dataset tailored to our distinct use case. Evaluating the fine-tuned model or a Retrieval Augmented Generation (RAG)-based model usually entails comparing its performance against a ground truth dataset if available. This aspect is crucial because the responsibility for ensuring the LLM performs as expected extends beyond the model itself; it becomes the engineer's responsibility to ensure that the LLM application generates the desired outputs. This task involves leveraging appropriate prompt templates, implementing effective data retrieval pipelines, considering the model architecture (if fine-tuning is required) and more. However, navigating the selection of the right components and conducting a comprehensive system evaluation remains a nuanced challenge.

We are going to consider evaluations not for building base LLMs, but for niche and specific use cases. We are going to assume that we are using a model and trying to build an application such as a chatbot for a bank, or a 'chat to your data' application. Further, we want our tests to be automatic and scalable to allow us to iteratively improve our application throughout the Machine Learning Operations (MLOPs) lifecycle! How do we build our own metrics?

### Automating Evaluations (Evals)

*What* should we evaluate?

1. **Context adherence**: (or groundedness) which is the question of whether the LLM response aligns with the provided context or guidelines.

2. **Context relevance**: for systems that use RAG, is the retrieved context relevant to the original query or prompt?

3. **Correctness**: (or accuracy) does the LLM output align with ground truth or the expected results?

4. **Bias and toxicity**: this is the negative potential that exists in LLMs that must be mitigated. First, bias is prejudice towards or away from certain groups. Toxicity can be considered harmful or implicit wording that is offensive to certain groups. 

*When* should we evaluate?

1. **After every change**: typically in software you test after every change whether that's bug fixes, feature updates, changes to the data, model or prompt template.

2. **Pre-deployment**: if testing after every change becomes too costly or slow, then periodic testing can be conducted more comprehensively throughout development. This could be when merging a branch to main, pushing code to production or more frequently at the end of each sprint.

3. **Post-deployment**: ongoing evaluation should be conducted and varies depending on the use case and business requirements. However, this is essential to mitigating drift and continually improving your model over time. Further, you should primarily focus on collecting user feedback and conducting A/B testing to iteratively enhance the user experience of your product.

### Rule Based Evaluations

One of the simplest things we may want to ensure when testing our generated responses is:

- Are there certain pieces of text that our system is expected to generate?

- Are there certain pieces of text that the system can not generate?

For our given application, the risk profile of the project will likely dictate that certain words can not be used under any circumstances. Furthermore, there may be certain words or phrases that we expect our system to generate. Here is an example of how we can test for this:

```python
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
    
    try:
      # Check if any of the expected words are present in the assistant's response
      assert any(word in answer.lower() \
                for word in expected_words), \
      # If the expected words are not found, raise an assertion error with a message
      f"Expected the assistant questions to include \
      '{expected_words}', but it did not"
    except Exception as e:
      print(f"An error occured: {str(e)}")

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
        human_template,
        system_message,
        llm,
        output_parser
    )
  
    # Invoke the assistant with the user's question
    answer = assistant.invoke({"question": question})
    # Print the assistant's response
    print(answer)
    
    try:
      # Check if the expected refusal is present in the assistant's response
      assert decline_response.lower() in answer.lower(), \
      # If the expected refusal is not found, raise an assertion error with a message
      f"Expected the bot to decline with '{decline_response}' got {answer}"
    except Exception as e:
      return(f"An error occured: {str(e)}")
      
```

These examples provided can serve as a template for creating rule-based evaluations for testing Large Language Models (LLMs). This approach can use regular expressions to test certain patterns. For example, you may wish to run a rule-based eval on:
- ranges of dates
- ensuring the system does not generate personal information (bank account numbers, unique identifiers etc.)

If you would like to design a rule-based eval. Use the following steps:
1. **Define Evaluation Function**: As in the examples provided, define evaluation functions tailored to specific evaluation criteria. These would take inputs like system messages, user questions and expected responses.
2. **Define Logic Check**: This could be exact matches, simple containment checks, or you could use regular expressions to define more complex patterns for expected responses. 
3. **Incorporate Logic Check into Assertions**: Modify the assertion checks in the evaluation function to use the defined logic for your use case, to reflect expected behaviour for your system.
4. **Documentation**: Include docstrings to describe the purpose of each evaluation function, the expected inputs and the evaluation criteria. The documentation helps maintain clarity and facilitates collaboration in your team.
5. **Error handling**: Implement error handling mechanisms within the evaluation functions to handle cases where the expected patterns are not found in the LLM's responses. This ensures failures are properly reported and diagnosed during testing.
6. **Parameterization**: Consider use of threshold values to allow you to adapt evaluation criteria depending on the risk profile of your use case. 

### Model-Graded Evaluations

How can we better ensure the overall quality of the generated responses in our system? We want to have an evaluation method that is more robust and comprehensive. This can be tricky as what defines a "good" response can be highly subjective. Rule-based evals can be useful to ensure certain data or information is or is not in our output. However, this gets more difficult and fragile as the application grows.

One method is to use an LLM to evaluate an LLM app, known as Model-Graded Evals. Yes, that's right, we are using AI to check AI.

In this code snippet, we are implementing a mechanism to evaluate the quality of the responses generated. Specifically, we are looking at the format of the response, although any feature of the generated text could be considered by adapting the `eval_system_prompt` and the `eval_user_message`. We need to define a `create_eval_chain` function which is the LLM that will be grading the main LLM application.

```python
def create_eval_chain(
    agent_response,
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    output_parser=StrOutputParser()
):
    """
    Creates an evaluation chain to assess the appropriateness of the agent's response.

    Parameters:
        agent_response (str): The response generated by the agent.
        llm (ChatOpenAI, optional): Language model used for evaluation.
            Defaults to ChatOpenAI(model="gpt-3.5-turbo", temperature=0).
        output_parser (OutputParser, optional): Output parser for parsing agent's response.
            Defaults to StrOutputParser().

    Returns:
        ChatPromptTemplate: Evaluation chain for assessing the agent's response.
    """

    delimiter = "####"

    eval_system_prompt = f"""You are an assistant that evaluates whether or not an assistant is producing valid responses.
    The assistant should be producing output in the format of [CUSTOM EVALUATION DEPENDING ON USE CASE]."""

    eval_user_message = f"""You are evaluating [CUSTOM EVALUATION DEPENDING ON USE CASE].
    Here is the data:
      [BEGIN DATA]
      ************
      [Response]: {agent_response}
      ************
      [END DATA]

    Read the response carefully and determine if it [MEETS REQUIREMENT FOR USE CASE]. Do not evaluate if the information is correct only evaluate if the data is in the expected format.

    Output 'True' if the response is appropriate, output 'False' if the response is not appropriate.
    """
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", eval_system_prompt),
        ("human", eval_user_message),
    ])

    return eval_prompt | llm | output_parser
```

The `create_eval_chain` function creates an evaluation chain, which is essentially a series of steps that our LLM will follow to assess a given response. This function takes the generated response as input, along with which model and output parser you would like to use for evaluation. 

We specify the evaluation model to output 'True' or 'False' as this would allow for easy interpretation as to if our model passes/fails a multitude of these tests. This makes the testing regime scalable.

As the LLM outputs a string, we need to do some post-processing on the test to obtain our test results in a useful format. The following function `model_grad_eval()` could be re-used for processing all our model-graded evals in a given test suite. 

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
    eval_chain = create_eval_chain(generated_output)

    # Retrieve the evaluation result from the evaluation chain
    evaluation_results_str = eval_chain.invoke({})

    # Strip leading/trailing whitespaces and convert to lowercase
    retrieval_test_response = evaluation_results_str.strip().lower()

    # Check if "true" or "false" is present in the response
    try:
      if "true" in retrieval_test_response:
          return True
      elif "false" in retrieval_test_response:
          return False
    except Exception as e:
      return(f"An error occured: {str(e)}")

model_grad_eval_format(generated_output="[UNDESIRED OUTPUT GENERATED BY AGENT]")
```

The `model_grad_eval_format()` function parses the evaluation result to return a boolean True or False or raises an exception. By using evals, we can systematically assess the quality of responses generated by language models, enabling us to identify areas for improvement and enhance the overall performance of our application.

Some further tips on model-based evals:

- **Opt for the most robust model you can afford**: Advanced reasoning capabilities are often required to effectively critique outputs. Your deployed system might need to have low latency for user experience. However, a slower and more powerful LLM might be needed for evaluating outputs effectively.

- **You could create a problem within a problem**: Without careful consideration and design of your model-based evals. The evaluating model might make errors and give you a false sense of confidence in your system.

- **Create positive and negative evals**: Something cannot be logically true and untrue at the same time. Carefully design your evals to increase confidence.


## Conclusion

- Effective evaluation of LLM-based applications requires a multi-faceted approach, encompassing both rule-based and model-graded evals to assess various aspects of system performance.

- You should continually refine and adapt both your system and eval methods iteratively to keep pace with evolving user expectations, system requirements and advancements in LLM technology.

- By embracing a systematic, iterative and observability-based approach to evaluation, stakeholders can gain valuable insights into the strengths and weaknesses of LLM-based applications, driving continuous improvement and innovation to LLM application development and deployment.

## Further Reading

Please see below a list of relevant resources to this blog post:

[OpenAI Evals Guide](https://cookbook.openai.com/examples/evaluation/getting_started_with_openai_evals)

[Jane Huang's (Microsoft) Blog post](https://medium.com/data-science-at-microsoft/evaluating-llm-systems-metrics-challenges-and-best-practices-664ac25be7e5)

[LlamaIndex Notebook on Evals](https://github.com/run-llama/ai-engineer-workshop/blob/main/notebooks/02_evaluation.ipynb)

[Hamel H.'s blog](https://hamel.dev/blog/posts/evals/#eval-systems-unlock-superpowers-for-free)

[LlamaIndex's Docs](https://docs.llamaindex.ai/en/v0.10.17/optimizing/evaluation/evaluation.html)

[Free course on automated testing for LLMOps with CircleCI](https://learn.deeplearning.ai/courses/automated-testing-llmops/lesson/1/introduction)