#  LangChain + LLaMA-2 Web Knowledge Retrieval System

---

##  Company Banners

<div align="center">
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="Hugging Face" height="60" style="margin-right:40px;"/>
  <img src="https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/langchain-stack.svg" alt="LangChain" height="60" style="margin-right:40px;"/>
  <img src="https://raw.githubusercontent.com/facebookresearch/llama/main/docs/llama2.png" alt="LLaMA 2" height="60"/>
</div>

---

##  Overview

This notebook demonstrates how to build a **Retrieval-Augmented Generation (RAG)** system using **LangChain**, **LLaMA-2**, and **FAISS**.
The model reads web articles about open-source LLMs (like Vicuna, MPT, and StableLM), embeds them, and uses LLaMA-2 to answer questions contextually.

You will:

* Scrape multiple blog URLs
* Split and embed content
* Store vectors using FAISS
* Query using a fine-tuned LLaMA-2 model

---

##  Step 1: Install Dependencies

Install all required libraries for text embedding, document processing, and model loading.

```python
!pip -q install langchain bitsandbytes accelerate transformers datasets loralib sentencepiece pypdf sentence_transformers
!pip -q install unstructured langchain_community tokenizers xformers
```

---

##  Step 2: Load Web Data

Use **UnstructuredURLLoader** to scrape articles about open LLMs.

```python
from langchain.document_loaders import UnstructuredURLLoader

URLs = [
    'https://blog.gopenai.com/paper-review-llama-2-open-foundation-and-fine-tuned-chat-models-23e539522acb',
    'https://www.mosaicml.com/blog/mpt-7b',
    'https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models',
    'https://lmsys.org/blog/2023-03-30-vicuna/'
]

loaders = UnstructuredURLLoader(urls=URLs)
data = loaders.load()
```

---

##  Step 3: Split Documents into Chunks

Split the scraped text into manageable segments for embedding.

```python
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(data)
```

---

##  Step 4: Embed Text Using Hugging Face

Use **HuggingFaceEmbeddings** to transform text chunks into vector representations.

```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()
query_result = embeddings.embed_query("How are you")
```

---

##  Step 5: Build a FAISS Vector Database

Store all embeddings in a **FAISS** vector index for efficient retrieval.

```python
!pip install faiss-cpu

from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(text_chunks, embeddings)
index_name = 'llama'
vectorstore.save_local(index_name)
```

---

##  Step 6: Initialize LLaMA-2 Model

Authenticate with Hugging Face, then load **LLaMA-2-7B-Chat** using Transformers.

```python
from huggingface_hub import notebook_login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

notebook_login()
model = "daryl149/llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model,
    device_map='auto',
    torch_dtype=torch.float16,
    use_auth_token=True,
    load_in_8bit=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_new_tokens=512,
    do_sample=True,
    top_k=30,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)
```

---

##  Step 7: Create the LLM Wrapper

Wrap the LLaMA-2 model into a **LangChain-compatible pipeline**.

```python
from langchain import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})
llm.predict("Please provide a concise summary of the Book Harry Potter")
```

---

##  Step 8: Build Retrieval Q&A Chain

Use **RetrievalQA** to combine the LLM and FAISS retriever for question answering.

```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

query = "How good is Vicuna?"
qa.run(query)
```

Ask follow-up questions interactively:

```python
import sys
while True:
  user_input = input("Input Prompt: ")
  if user_input == 'exit':
    print('Exiting')
    sys.exit()
  if user_input == '':
    continue
  result = qa({'query': user_input})
  print(f"Answer: {result['result']}")
```

---

##  Example Queries

* “How does LLaMA-2 outperform other models?”
* “What are the main improvements in Vicuna?”
* “Summarize the StableLM architecture.”

---

##  Summary

This project demonstrates:

1. **Automated web data ingestion** using UnstructuredURLLoader
2. **Document vectorization** with Hugging Face embeddings
3. **Retrieval and response generation** using LLaMA-2
4. **Persistent FAISS index** for efficient query search

---

**Author:** [Chiejina Chike Obinna](https://github.com/obinnachike)
**Frameworks:** LangChain • Hugging Face • LLaMA-2 • FAISS
