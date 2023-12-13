# 📘 LLM Distillation Playbook: 12 Best Practices from Real Experiments

## 🚀 Get Started

### Open Source

Fine-tune LLMs like a pro: Our open-source Ludwig makes it easy to adapt LLMs to your specific needs, all in a YAML file. No need to code everything from scratch! Check out our [getting started guide](https://ludwig.ai/latest/getting_started/llm_finetuning/). We provide sample configs under `configs/` to help you kick off experiments quickly.

Serve them at scale: LoRaX handles serving thousands of LLMs simultaneously, effortlessly. Check out our GitHub repository: [LoRaX](https://github.com/predibase/lorax)

### Fully-Managed

Get started in minutes: No need to build everything yourself. Our managed platform provides everything you need to train and serve LLMs with ease. Sign up for a [free trial](https://predibase.com/)!

🚀 Start distilling LLMs today with our comprehensive playbook and tools! 🧪✨

## A Practical Guide to Distilling LLMs

- 📊 **Data Quality is King:** The quality of your training dataset is paramount. We saw significant performance improvements with every step up in data quality.

- 📏 **Quantity Matters, but Quality Rules:** While larger datasets offer performance boosts, a smaller, high-quality dataset outperforms a large, lower-quality one.

- 🪄 **Hyperparameters do matter. Here's a quick guide to quickly find them so you can focus on data:**
  - ⚙️ **Max out your GPU:** The larger the batch size, the faster your training.
  - 🔄 **Learning Rate:** Experiment between 1e-5 and 5e-4. Cosine decay is a safe bet.
  - 🎯 **Small Batch Size? No Problem:** Gradient accumulation reduces noise from small mini batches. It's important to adjust the learning rate. Doing it proportionally to (batch size * gradient accumulation steps) is a rule of thumb I follow.

### Guide to Your First Open Source LLM When You Don't Have Data:

- 👢 **Bootstrap your data:** Use a closed-source LLM (where allowed) to get started.
- 💡 **Fine-tune an open-source LLM:** This fine-tuned model will pack a punch.
- 📈 **Boost your data:** Refine your dataset by manually reviewing, dropping bad examples, and adding new ones. This will push your LLM beyond the closed-source model.

## LLM Fine-Tuning Best Practices

The need for smaller, efficient versions of large language models (LLMs) is rising rapidly. Despite their impressive capabilities, LLMs can be costly, resource-intensive, and slow. Distilling these models into smaller, more efficient counterparts offers a compelling solution, balancing capability with cost and speed.

### 1. Understand the limitations of smaller models.
- Despite being a popular technique, model distillation is not guaranteed to work well in all cases, which depends on the task and data.

### 2. Build good logging infrastructure.
- Have basic logging infrastructure for teacher models in production. If logs are limited due to low traffic, PII, or other constraints, purely synthetic data generation may be a viable option.

### 3. Define clear evaluation criteria.
- Effective evaluation of distilled models requires clearly defined criteria that align with your specific application's needs. The choice of evaluation metrics should reflect the nature of the problem and the desired outcomes of the model.

### 4. Maximize the quality of your teacher model.
- The quality of your teacher model's outputs serves as an upper limit for the performance of your distilled student model. Invest in maximizing the quality of your teacher model's performance as much as possible.

### 5. Use auxiliary techniques to maximize data quality offline.
- To give your student models an edge, consider how you might fundamentally improve the quality of your data with manual labeling, rules-based filtering, auxiliary model ranking, or LLM chaining.

### 6. The best datasets are diverse and balanced.
- Try to make your dataset as diverse, non-repetitive, and balanced as you can. The more scenarios and complexities your dataset covers, the more likely the distilled student will generalize and be unbiased.

### 7. Start small. No, smaller.
- Start with smaller, simpler model configurations that are quick to train so that you can debug issues with your setup, iterate quickly, and establish good benchmarks for comparing to more complex model configurations later.

### 8. Assess the marginal utility of having more data.
- To answer the question of how much data do you need for fine-tuning, run an ablation of varying dataset size and extrapolate.

### 9. Consider how you want to serve your fine-tuned models.
- While not crucial to decide upfront, have a model serving plan in mind to prioritize experiments with models that can ultimately be served.

### 10. Experiment broadly, one parameter at a time.
- Exploration over exploitation: spend most of your time and energy to gain insight into the problem. Change one variable at a time, and try not to rathole.

### 11. Actually look at the model’s mistakes.
- While aggregate metrics and advanced automated evaluation methods provide a broad overview of model performance, there is unparalleled value in manually reviewing individual examples of your model's outputs.

### 12. Monitor your models in production and A/B test them with real users.
- While test sets provide a controlled environment for evaluation, the true test of your model’s effectiveness is how it performs with actual users and real-time inputs. Deploy your model and observe its performance in a real-world setting!
