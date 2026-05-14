import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core import vectorstores
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import RetrievalQA   #这里之前是老的版本  现在被修改了
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import DashScopeEmbeddings

# from langchain.chains import create_stuff_documents_chain
# from langchain.chains import RetrievalQA


# 安装的langchain版本比较高，新版本中不是这样调用的，稍等我改下试试
# 好的  我没注意版本 自动安装的
# 这样你再试试应该就行了 ，或者降级下langchain版本到0.2.x，代码写的版本比较老
# OK  那我就督纳克i了嗯好的  嗯嗯 有什么问题你在发群里，班主任老师对接  好
#------- 基于今天讲解的langchain 的框架，开发对本地知识库进行问答的逻辑，只需要包括文档检索 + llm回答流程(参考项目2)；

load_dotenv()

# DASHSCOPE_API_KEY="sk-0c18954972f8450c95480d5c3c0a82f6"
# DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"


api_key = "sk-0c18954972f8450c95480d5c3c0a82f6"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# if not api_key or not base_url:
#     raise ValueError("请先设置 DASHSCOPE_API_KEY 和 DASHSCOPE_BASE_URL")

llm = ChatOpenAI(
    model="qwen-flash",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-0c18954972f8450c95480d5c3c0a82f6",
    temperature=0,
)


#读文件

loader = PyPDFLoader("汽车知识手册.pdf")
docs = loader.load()

embeddings = DashScopeEmbeddings(model="text-embedding-v2", dashscope_api_key=api_key)
#切
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=1)
#split = text_splitter.split_text(docs)
splits = text_splitter.split_documents(docs) # 正确
#存向量
#vector_store = FAISS().from_documents(docs)
vector_store = FAISS.from_documents(docs, embedding=embeddings) # 注意：这里可能需要传入 embedding 模型
#检索
retriever = vector_store.as_retriever()
#context = retriever.invoke("汽车保养周期是多久")


# ============================================================
# 写法一：ChatPromptTemplate（推荐）
# ============================================================
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","你是一个专业的汽车行业的专家,回答用户提问的问题。如果资料中没有，请回答'我不知道'。\n\n参考资料：{context}"),
        ("human","我想咨询一下：{question} "),
    ]
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,                     # 你的通义千问大模型
    retriever=retriever,         # 你的 FAISS 向量库检索器
    chain_type="stuff",          # 默认模式，把所有检索到的文档塞进提示词里
    chain_type_kwargs={"prompt": prompt}  # 传入你刚刚自定义好的 Prompt 模板
)

question = "汽车的保养周期是多久？"
result = qa_chain({"query": question})

print("AI的回答：", result["result"])



