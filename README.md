# LLM Distillation Playbook

**Justin Zhao<sup>&dagger;</sup>, Wael Abid<sup>&dagger;</sup>**

&dagger; Predibase, MLX team

Inspired by the [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook).

## Table of Contents

-   [Who is this document for?](#who-is-this-document-for)
-   [Why a distillation playbook?](#why-a-distillation-playbook)
-   [Commitment to open source](#commitment-to-open-source)
-   [Best practices](#best-practices)

## Who is this document for?

This document is for engineers and ML practitioners interested in **LLM distillation**. We assume basic knowledge of language models and deep learning concepts.

Our emphasis is on the **model distillation** and the efficient deployment of adapter-based fine-tuned open source models.

## Why a distillation playbook?

As large language models (LLMs) become increasingly capable and integral to various applications, the need for their efficient, smaller counterparts has never been more pronounced. This shift is driven by the compelling performance of LLMs, juxtaposed with the significant costs, resource demands, and slower operational speeds of large models. In response, distilling these models into more efficient, smaller versions presents a solution that balances capability with cost-effectiveness and speed.

Despite significant interest in model distillation, we find that the state of the world that inspired the [tuning playbook](https://github.com/google-research/tuning_playbook/) has hardly changed. There is *still* an astonshing amount of toil and guesswork involved in actually getting deep neural networks to work well in practice, and this has continued onto the world of LLMs. Anecdotes and snippets of advice are spread across arxiv, huggingface, discord, substack, and social media, but how well these recommendations are centralized and systematized remains to be seen.

The advice in this document draws from our experience distilling language models at Google and Predibase, combined with any LLM research and media we could find on the topic to date. We are hopeful that these strategies for the efficient refinement of LLMs provide practitioners and enthusiasts with ideas that are practical, grounded in academic research, and helpful for the growing development and utilization of open source language models.

## Commmitment to open source

At Predibase, we believe that the future is fine-tuned, specialized, and open source LLMs. Open source is in the DNA of the company, and as a company we maintain:

- [Ludwig](https://github.com/ludwig-ai/ludwig): Low-code framework for building custom LLMs, neural networks, and other AI models
- [LoRAX](https://github.com/predibase/lorax): Multi-LoRA inference server that scales to 1000s of fine-tuned LLMs
- [Horovod](https://github.com/horovod/horovod): Distributed training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.

Our managed platform is built on top of these repositories, and you can sign up for a free trial [here](https://predibase.com/).

## Best practices

### 1. Understand the limitations of smaller models.

Despite being a popular technique, model distillation is not guaranteed to work well in all cases, which depends on the task and data.

### 2. Build good logging infrastructure.

Have basic logging infrastructure for teacher models in production. If logs are limited due to low traffic, PII, or other constraints, purely synthetic data generation may be a viable option.

### 3. Define clear evaluation criteria.

Effective evaluation of distilled models requires clearly defined criteria that align with your specific application's needs. The choice of evaluation metrics should reflect the nature of the problem and the desired outcomes of the model.

### 4. Maximize the quality of your teacher model.

The quality of your teacher model's outputs serves as an upper limit for the performance of your distilled student model. Invest in maximizing the quality of your teacher model's performance as much as possible.

### 5. Use auxiliary techniques to maximize data quality offline.

To give your student models an edge, consider how you might fundamentally improve the quality of your data with manual labeling, rules-based filtering, auxiliary model ranking, or LLM chaining.

### 6. The best datasets are diverse and balanced.

Try to make your dataset as diverse, non-repetitive, and balanced as you can. The more scenarios and complexities your dataset covers, the more likely the distilled student will generalize and be unbiased.

### 7. Start small. No, smaller.

Start with smaller, simpler model configurations that are quick to train so that you can debug issues with your setup, iterate quickly, and establish good benchmarks for comparing to more complex model configurations later.

### 8. Assess the marginal utility of having more data.

To answer the question of how much data do you need for fine-tuning, run an ablation of varying dataset size and extrapolate.

### 9. Consider how you want to serve your fine-tuned models.

While not crucial to decide upfront, have a model serving plan in mind to prioritize experiments with models that can ultimately be served.

### 10. Experiment broadly, one parameter at a time.

Exploration over exploitation: spend most of your time and energy to gain insight into the problem. Change one variable at a time, and try not to rathole.

### 11. Actually look at the model’s mistakes.

While aggregate metrics and advanced automated evaluation methods provide a broad overview of model performance, there is unparalleled value in manually reviewing individual examples of your model's outputs.

### 12. Monitor your models in production and A/B test them with real users.

While test sets provide a controlled environment for evaluation, the true test of your model’s effectiveness is how it performs with actual users and real-time inputs. Deploy your model and observe its performance in a real-world setting!
