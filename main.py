import os
from collections import namedtuple
from typing import List

from dotenv import load_dotenv
from huggingface_hub import HfApi
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from markitdown import MarkItDown
from smolagents import CodeAgent, HfApiModel

from retriever_tool import RetrieverTool

load_dotenv()

ConvertToMarkdownResult = namedtuple("ConvertToMarkdownResult", ["title", "content"])


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_and_convert_to_markdown(file_path: str) -> ConvertToMarkdownResult:
    md = MarkItDown()
    result = md.convert(file_path)
    return ConvertToMarkdownResult(title=result.title, content=result.text_content)


def create_document(file_path: str) -> Document:
    convert_to_markdown_result = read_and_convert_to_markdown(file_path)

    metadata = {
        "source": os.path.basename(file_path),
        "title": convert_to_markdown_result.title,
    }

    return Document(page_content=convert_to_markdown_result.content, metadata=metadata)


def load_documents() -> List[Document]:
    documents = []

    folder_path = "./docs"
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            documents.append(create_document(file_path))

    return documents


def main():
    source_docs: List[Document] = load_documents()
    print(source_docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs_processed = text_splitter.split_documents(source_docs)

    retriever_tool = RetrieverTool(docs_processed)

    agent = CodeAgent(
        tools=[retriever_tool],
        model=HfApiModel("meta-llama/Meta-Llama-3-8B-Instruct"),
        max_iterations=4,
        verbose=True,
    )

    agent_output = agent.run("What is my name?")

    print("Final outupt:")
    print(agent_output)


if __name__ == "__main__":
    main()
