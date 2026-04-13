import os
# Loaders
from langchain_community.document_loaders import DirectoryLoader, WikipediaLoader, PythonLoader
from langchain_openai import ChatOpenAI
# Splitters
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    PythonCodeTextSplitter,
)

# Core (NEW structure)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel

# -----------------------------
# 1. MULTI-SOURCE LOADING
# -----------------------------
def load_documents():
    docs = []

    # Load .txt files
    dir_loader = DirectoryLoader("./data", glob="*.txt")
    docs.extend(dir_loader.load())

    # Load Wikipedia
    wiki_loader = WikipediaLoader(query="Artificial Intelligence", load_max_docs=1)
    docs.extend(wiki_loader.load())

    # Load Python files
    py_loader = DirectoryLoader("./data", glob="*.py", loader_cls=PythonLoader)
    docs.extend(py_loader.load())

    # Print metadata
    print("\n=== Loaded Documents Metadata ===")
    for doc in docs:
        print({
            "source": doc.metadata.get("source"),
            "file_path": doc.metadata.get("file_path"),
        })

    return docs


# -----------------------------
# 2. ADAPTIVE SPLITTING
# -----------------------------
def split_by_type(doc: Document):
    source = doc.metadata.get("source", "")

    if source.endswith(".md"):
        print(f"Using Markdown splitter for {source}")
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header")])
        return splitter.split_text(doc.page_content)

    elif source.endswith(".py"):
        print(f"Using Python splitter for {source}")
        splitter = PythonCodeTextSplitter()
        chunks = splitter.split_text(doc.page_content)
        return [Document(page_content=c, metadata=doc.metadata) for c in chunks]

    else:
        print(f"Using Recursive splitter for {source}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(doc.page_content)
        return [Document(page_content=c, metadata=doc.metadata) for c in chunks]


# -----------------------------
# 3. CHAINS
# -----------------------------
# LLM (OpenAI)
llm = ChatOpenAI(
    base_url="http://192.168.159.1:1611/v1",
    api_key="dummy",
    model="liquid/lfm2.5-1.2b",
    temperature=0
)

# Summarise chain
summarise_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarise in exactly 3 bullet points"),
    ("human", "{text}")
])

summarise_chain = summarise_prompt | llm | StrOutputParser()

# Extract chain
extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract all proper nouns, dates, and numbers as a JSON list"),
    ("human", "{text}")
])

extract_chain = extract_prompt | llm | JsonOutputParser()


# -----------------------------
# 6. FALLBACK CHAINS
# -----------------------------
fallback_summary = RunnableLambda(lambda x: "Fallback summary: Unable to process input")
fallback_extract = RunnableLambda(lambda x: {"error": "Fallback extraction triggered"})

summarise_chain = summarise_chain.with_fallbacks([fallback_summary])
extract_chain = extract_chain.with_fallbacks([fallback_extract])


# -----------------------------
# 4. ROUTER
# -----------------------------
def route(input_data):
    if input_data["mode"] == "summarise":
        return summarise_chain
    else:
        return extract_chain

router = RunnableLambda(route)


# -----------------------------
# 5. PARALLEL MERGE
# -----------------------------
parallel_chain = RunnableParallel(
    summary=summarise_chain,
    entities=extract_chain
)


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def main():
    docs = load_documents()

    all_chunks = []
    for doc in docs:
        chunks = split_by_type(doc)
        all_chunks.extend(chunks)

    print("\n=== Processing Chunks ===")

    for i, chunk in enumerate(all_chunks[:3]):  # limit for demo
        text = chunk.page_content

        print(f"\n--- Chunk {i+1} ---")

        # Router test
        result_summary = router.invoke({"mode": "summarise", "text": text})
        print("\n[Router - Summary]")
        print(result_summary)

        result_extract = router.invoke({"mode": "extract", "text": text})
        print("\n[Router - Extract]")
        print(result_extract)

        # Parallel execution
        merged = parallel_chain.invoke({"text": text})
        print("\n[Parallel Merged Output]")
        print(merged)


if __name__ == "__main__":
    main()