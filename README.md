# 📘 Assignment 2: Branching Research Chain with Routing

## 🚀 Objective

Load documents from multiple sources, apply adaptive splitting, and route them through different LCEL sub-chains based on content type. No vector store is required.

---

## 🧱 Architecture Overview

* Multi-source document loading
* Adaptive text splitting based on file type
* Two independent LLM chains:

  * Summarisation
  * Entity Extraction
* Routing using RunnableLambda
* Parallel execution using RunnableParallel
* Fallback handling for error resilience

---

## 📂 Project Structure

```
assignment2.py
data/
  sample.txt
  greet.py
```

---

## 📥 Data Sources

* 📄 Local `.txt` files → DirectoryLoader
* 🌐 Wikipedia → WikipediaLoader
* 🐍 Python files → PythonLoader

---

## ✂️ Adaptive Splitting

| File Type               | Splitter Used                  |
| ----------------------- | ------------------------------ |
| `.md`                   | MarkdownHeaderTextSplitter     |
| `.py`                   | PythonCodeTextSplitter         |
| Others (`.txt`, `.pdf`) | RecursiveCharacterTextSplitter |

---

## 🔗 Chains

### 🟢 Summarisation Chain

* Prompt: Summarise in exactly 3 bullet points
* Output: String

### 🟣 Extraction Chain

* Prompt: Extract proper nouns, dates, numbers
* Output: JSON

---

## 🔀 Routing Logic

```python
def route(input_data):
    if input_data["mode"] == "summarise":
        return summarise_chain
    else:
        return extract_chain
```

---

## ⚡ Parallel Execution

```python
RunnableParallel(
    summary=summarise_chain,
    entities=extract_chain
)
```

---

## 🛡️ Fallback Handling

Each chain is wrapped with fallback:

* Summary fallback → returns default message
* Extraction fallback → returns error JSON

### Why Fallback?

* Prevents crashes
* Handles LLM/API failures
* Ensures stable output

---

## 🧪 Sample Output

```
[Router - Summary]
• AI enables machines to learn from data
• Machine Learning improves performance
• Deep Learning uses neural networks

[Router - Extract]
{
  "proper_nouns": ["Artificial Intelligence", "Machine Learning"],
  "dates": ["2023"],
  "numbers": []
}
```

---

## ⚙️ Setup Instructions

### Install dependencies

```bash
pip install langchain-core langchain-community langchain-openai langchain-text-splitters wikipedia
```

---

### Set API / LLM

```python
llm = ChatOpenAI(
    base_url="your Url",
    api_key="dummy",
    model="your-model",
    temperature=0
)
```

---

### Run the project

```bash
python assignment2.py
```

---

## ✅ Features Implemented

* Multi-source loading
* Adaptive splitting
* Routing with RunnableLambda
* Parallel execution
* Fallback error handling
* Clean structured output

---

## 🎯 Conclusion

This project demonstrates how to build a robust and modular LLM pipeline using LangChain’s LCEL framework with routing, branching, and fallback handling.

---

## 📊 Output Screenshot

![Output](images/Output1.png)

![Output](images/output2.png)

---

## 👨‍💻 Author

Manohara A R
