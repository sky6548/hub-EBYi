import yaml
import os
import glob
import datetime
import pdfplumber
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


class LocalRAG:
    def __init__(self):
        self.embedding_model = SentenceTransformer(config["embedding"]["model_path"])
        self.embedding_dims = config["embedding"]["dimensions"]
        self.chunk_size = config["chunk"]["chunk_size"]
        self.chunk_overlap = config["chunk"]["chunk_overlap"]

        self.llm = ChatOpenAI(
            model=config["llm"]["model_name"],
            api_key=config["llm"]["api_key"],
            base_url=config["llm"]["base_url"],
            temperature=0.1,
            top_p=0.9
        )

    def load_documents(self, directory: str) -> List[Document]:
        docs = []
        for file_path in glob.glob(os.path.join(directory, "**/*"), recursive=True):
            if file_path.endswith(".pdf"):
                docs.extend(self._load_pdf(file_path))
        return docs

    def _load_pdf(self, file_path: str) -> List[Document]:
        documents = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file_path, "page": page_num + 1}
                    ))
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        return splitter.split_documents(documents)

    def create_vectorstore(self, documents: List[Document], persist_dir: str):
        embeddings = HuggingFaceEmbeddings(
            model_name=config["embedding"]["model_path"],
            model_kwargs={"device": config["device"]}
        )
        return Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir
        )

    def load_vectorstore(self, persist_dir: str):
        embeddings = HuggingFaceEmbeddings(
            model_name=config["embedding"]["model_path"],
            model_kwargs={"device": config["device"]}
        )
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    def get_retriever(self, vectorstore, top_k: int = 5):
        return vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )

    def get_rerank_retriever(self, vectorstore, query: str, top_k: int = 5):
        docs = vectorstore.similarity_search(query, k=top_k * 2)
        if config["rerank"]["enabled"]:
            reranker = CrossEncoder("BAAI/bge-reranker-base")
            scored = reranker.rank(query, [doc.page_content for doc in docs], return_scores=True)
            scored.sort(key=lambda x: x["corpus_id"])
            return [docs[s["corpus_id"]] for s in scored[:top_k]]
        return docs[:top_k]

    def build_rag_chain(self, retriever):
        prompt = PromptTemplate.from_template(
            "现在的时间是{#TIME#}。你是一个专家，你擅长回答用户提问，帮我结合给定的资料，回答下面的问题。\n"
            "如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。如果提问不符合逻辑，请回答无法回答。\n"
            "如果问题可以从资料中获得，则请逐步回答。\n\n"
            "资料：\n{#CONTEXT#}\n\n"
            "问题：{#QUESTION#}"
        )

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        class RAGChain:
            def __init__(self, retriever, prompt, format_docs, llm):
                self.retriever = retriever
                self.prompt = prompt
                self.format_docs = format_docs
                self.llm = llm

            def invoke(self, question: str) -> str:
                docs = self.retriever.invoke(question)
                context = self.format_docs(docs)
                prompt_text = self.prompt.format(
                    TIME=datetime.datetime.now(),
                    CONTEXT=context,
                    QUESTION=question
                )
                messages = [HumanMessage(content=prompt_text)]
                return self.llm.invoke(messages).content

        return RAGChain(retriever, prompt, format_docs, self.llm)

    def chat(self, question: str, use_rerank: bool = False) -> str:
        persist_dir = config["vectorstore"]["persist_directory"]
        vectorstore = self.load_vectorstore(persist_dir)

        if use_rerank:
            retriever = lambda q: self.get_rerank_retriever(vectorstore, q, config["rerank"]["top_k"])
        else:
            retriever = vectorstore.as_retriever(search_kwargs={"k": config["rerank"]["top_k"]})

        rag_chain = self.build_rag_chain(retriever)
        return rag_chain.invoke(question)


if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["index", "query"], default="query")
    parser.add_argument("--docs_dir", default="./docs")
    parser.add_argument("--question", default="物业费如何计算？")
    args = parser.parse_args()

    rag = LocalRAG()

    if args.mode == "index":
        print("Loading documents...")
        docs = rag.load_documents(args.docs_dir)
        print(f"Loaded {len(docs)} documents")

        print("Splitting documents...")
        chunks = rag.split_documents(docs)
        print(f"Created {len(chunks)} chunks")

        print("Creating vectorstore...")
        rag.create_vectorstore(chunks, config["vectorstore"]["persist_directory"])
        print("Vectorstore created successfully")

    elif args.mode == "query":
        pprint.pprint(rag.chat(args.question, use_rerank=config["rerank"]["enabled"]))
