# Q&A Chatbot with RAG & VectorDB

## Task Description

- 특정 분야에 대한 데이터를 crawling 한 뒤, vectorDB 내에 데이터를 정제하여 삽입하고, vectorDB를 활용한 RAG (LLM; OpenAI) 기반으로 해당 데이터에 대한 Q&A를 할 수 있는 Chatbot을 구축.

## Requirements

- Specifying the reference code repository is mandatory.
- Implementing a Chatbot Interface for testing is required (e.g., Gradio / Streamlit / Langchain-Chat).
- Among the mentioned points, "Literature Review/Experimental Design/Result" as a report (README.md) should include at least one. (DO NOT TRY TO SOLVE TOO MANY THINGS; THOUGHTFULLY CONSIDERING & SOLVING JUST ONE POINT IS SUFFICIENT)
- Code Instructions must be included (reproducible/testable/readable)
- Describe your environment settings (environment guide; e.g. Dockerfile), structure of your code, and usage instructions

(1) Crawling & Chunking & Indexing : "Select one of the websites from three domains → Choose information on 'specific areas' within the site to crawl → Ingest structured information into vectorDB.”

- Point: How to crawl and what factors should you keep in mind when handling the gathered data?
- website domain candidates : Reddit, Stackoverflow, Stackexchange
- Reference : https://www.geeksforgeeks.org/scraping-reddit-using-python/

- Point : How to deal with structured information (e.g. Knowledge Base, List, Columns, Equation, Table, Markdown format, Image, Threads) to create an embedding vector for it
- Point : Chunking Strategy

(2) Q&A Pipeline & Prompt Engineering for Chunking/Indexing & Response Generation

- one of Config params : OpenAI API-KEY

- Point : How to evaluate the performance of your prompts
- Point : Explain the reasons behind why you design your Q&A pipeline

- Reference : https://github.com/wandb/wandbot

(3) Chatbot Evaluation

- Point : How to evaluate the E2E performance of your chatbot

- Reference : https://github.com/wandb/wandbot

## Report

The "Report" section outlines the necessary steps and configurations for setting up the Q&A Chatbot with RAG & VectorDB, along with references and a brief guide on how to execute the code. Let me elaborate on each part:

### 1. Crawling & Chunking & Indexing:

#### 1.1 Website Domain Selection and Crawling:

For the crawling task, we leverage the StackExchange API to collect Q&A data. In particular, we focus on the "law" category to gather questions, answers,( and comments). Additionally, we extract tag information for further analysis.

To obtain the required API key, users are advised to register a new application at Stack Apps OAuth registration, providing the following details:
OAUTH Domain: stackexchange.com
Application Website: https://stackapps.com
This approach ensures that our data collection is site-specific, tailored to the legal domain within StackExchange, allowing for a more targeted and relevant dataset for our Q&A Chatbot.
Details are in "./notebook/01_crawling.ipynb".

#### 1.2 Handling Structured Information:

In the case of StackExchange, the majority of the data is formatted in markup language, and there is a significant presence of link information. While exploring the option of extracting information by following links presented a challenge due to its complexity and resource-intensive nature, we opted to refine the data, acknowledging the difficulties in data preprocessing and preliminary analysis.

Various attempts were made, including experimenting with a MarkdownTextSplitter to remove markup language tags. However, it was observed that the existing implementation did not effectively eliminate these tags. Additionally, attempting summarization using ChatGPT posed challenges due to the substantial computational cost associated with the process, leading us to reconsider the strategy.

The task of creating data that enables LLMs to comprehend information across various multimodal formats (Knowledge Base, List, Columns, Equation, Table, Markdown format, Image, Threads) is likely to require multiple experiments. Given that LLMs are language-based, dealing with information like knowledge graphs may involve expressing it in natural language or organizing graph information through graph linearization to construct context. The way multimodal data is input and learned by LLMs may vary, influencing the approach to incorporating such data. For images, it could be beneficial to include a module that provides descriptions for the images.

#### 1.3 Chunking Strategy

Given these challenges, we adopted a Naive approach. The strategy involved chunking the raw data based on character length, as a practical and effective method to structure the information for the RAG database. Although this approach may seem straightforward, it proved to be a pragmatic solution under the circumstances, allowing us to progress in building the RAG database without compromising on data quality.

Further insights into this process are detailed in the provided notebook "./notebook/02_04_chunking_indexing_evaluation.ipynb."
.

### 2. Q&A Pipeline & Prompt Engineering:

#### 2.1 Prompt Performance Evaluation:

Firstly, the fundamental strategy involves the Eyeballing approach, where a QA validation set is created for the chatbot. This set is then utilized as a dataset to evaluate how well the prompt chatbot aligns with the evaluation set, using qualitative evaluation metrics such as exact match, BLUE, METEOR, and others. It is crucial to refine the dataset meticulously by selecting better samples based on user or expert feedback for the evaluation set.

The WandbBot's evaluation strategy [part1](https://wandb.ai/wandbot/wandbot-eval/reports/How-to-evaluate-an-LLM-Part-1-Building-an-Evaluation-Dataset-for-our-LLM-System--Vmlldzo1NTAwNTcy) provides detailed explanations on this approach, and I have incorporated this information to create a simple evaluation in the notebook.

#### 2.2 Q&A Pipeline Design:

This question is related to the purpose of using the RAG system. When relying solely on a simple LLM, issues arise regarding the inability to utilize context effectively and adapt to specific situations. LLM itself struggles with context comprehension, is heavily dependent on training data, and tends to generate responses biased towards the priors in the training data. Additionally, it faces challenges in handling various multimodal data. To address these issues, the RAG system was introduced, aiming to enhance the LLM's understanding of questions by retrieving relevant information.

An essential aspect here is how to organize and retrieve the data. As the amount of context in prompts increases, there is a problem of escalating computational complexity due to the growing token size. To mitigate this, a strategy involves summarizing the content to reduce the amount of context. Moreover, as described in part1, there is a need to refine and filter out duplicate samples that are closely distributed in the embedding space.

### 3. Chatbot Evaluation:

3.1 E2E Performance Evaluation:
The overall evaluation strategy is thoroughly introduced in WandbBot's evaluation strategy [eval](https://wandb.ai/ayush-thakur/llm-eval-sweep/reports/Evaluating-LLM-Systems-and-Hyperparameter-Optimization--Vmlldzo0NzgyMTQz) [part1](https://wandb.ai/wandbot/wandbot-eval/reports/How-to-evaluate-an-LLM-Part-1-Building-an-Evaluation-Dataset-for-our-LLM-System--Vmlldzo1NTAwNTcy), [part2](https://wandb.ai/wandbot/wandbot-eval/reports/How-to-evaluate-an-LLM-Part-2-Manual-Evaluation-of-our-LLM-System--Vmlldzo1NzU4NTM3), [part3](https://wandb.ai/wandbot/wandbot-eval/reports/How-to-evaluate-an-LLM-Part-2-Manual-Evaluation-of-our-LLM-System--Vmlldzo1NzU4NTM3).

With this information, I haven't had the chance to implement the entire system, but there seem to be some good points in the general chatbot evaluation strategy.

Part1: starts by understanding the distribution of user queries and proceeds to preprocessing these queries. The consideration of token counting for filtering, understanding question semantics, and the removal of near-duplicates are crucial steps. Clustering similar questions is explored, and GPT-4 is used to sample from these clusters.

Part2: Manual evaluation serves as a gold standard, and the subsequent section focuses on the RAG LLM Evaluation Dataset. It delves into what constitutes an accurate LLM response and provides insights into result analysis from manual evaluation. The evaluation metrics include the accuracy of Wandbot's responses, meta metrics like link hallucination and query relevancy, and the use of Argilla as a manual annotation tool.

Part3: The Typical RAG Pipeline is explained, emphasizing the evaluation of response quality, faithfulness, and context relevance.

---

### Additional Information:

#### API Key setting:

The API key should be entered in the .env.template file with your personal API key, and then remove the .template extension.

OPENAI_API_KEY: To obtain a ChatGPT API key, visit the [OpenAI website](platform.openai.com), sign in or create an account, navigate to the API section, and follow the provided instructions to generate your API key.

LLAMA_API_KEY: To obtain a LLAMA API key, visit the [Llama website](https://ai.meta.com/llama/), sign in or create an account, navigate to the API section, and follow the provided instructions to generate your API key.

STACKEXCHANGE_API_KEY: obtain by registering a new application at https://stackapps.com/apps/oauth/register. Enter the following information to obtain the KEY:

OAUTH Domain: stackexchange.com
Application Website: https://stackapps.com

#### notebook

The "notebook" directory contains a file with information related to crawling, chunking, indexing, and evaluation. These notebooks are crucial for understanding the implementation details of the chatbot.
The code in the notebook needs to be executed to generate the vector database before the chatbot can be run.

#### Q&A Chatbot execution commmand

After setting up the Docker environment, you can execute the following command to run the chatbot.

```bash
python3 chatbot.py
```

visit http://127.0.0.1:7860/ with your web browser.

## References

Building a Multi-User Chatbot with Langchain and Pinecone

- Tools : Pinecone, OpenAI, Ably, CockroachDB, Fingerprint Pro
- Building a Multi-User Chatbot with Langchain and Pinecone in Next.JS

Q&A Bot for W&B Documentation

- Creating a Q&A Bot for W&B Documentation

WandBot: GPT-4 Powered Chat Support

- Tools : llama-index, OpenAI, Langchain, wandb, FAISS, FastAPI
- Note : Utilizing HydeEmbeddings
- WandBot: GPT-4 Powered Chat Support
- https://github.com/wandb/wandbot

Building QnA system with Langchain, Pinecone, and LLMs

- Building a Document-based Question Answering System with LangChain, Pinecone, and LLMs like GPT-4.
