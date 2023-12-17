from dotenv import load_dotenv
load_dotenv()
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import Chroma

# Embedings
from langchain.embeddings import OpenAIEmbeddings 
from langchain.embeddings import LlamaCppEmbeddings

from langchain.chat_models import ChatOpenAI 
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.llms import CTransformers
from langchain.embeddings import HuggingFaceEmbeddings

# loader
loader = PyPDFLoader('unsu.pdf')
pages = loader.load_and_split()

# split
text_spliter = RecursiveCharacterTextSplitter(
    # set a really small chunk size, just to show
    chunk_size = 300, # 100글자 단위로 쪼갠다.
    chunk_overlap = 20,
    length_function = len, # python len function
    is_separator_regex = False, # 정규 표현식 regex False
)

texts = text_spliter.split_documents(pages)

llm_llama = CTransformers(
    model = 'llama-2-7b-chat.ggmlv3.q3_K_L.bin',
    model_type = 'llama'
)

# # embeding
# # embeddings_model = OpenAIEmbeddings() # 여길 유료로 전환해야한다.
embeddings_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

print('done')




# # load it into Chroma 
# db = Chroma.from_documents(texts, embeddings_model)

# # MultiQuery Retriever
# # Question
# question = '아내가 먹고 싶어하는 음식은 무엇이야?'
# llm = ChatOpenAI(temperature=0) # 무작위성을 얼마나? 0 --> 비슷비슷한 일반적인 대답 / , max_tokens=4095
# llm_llama2 = CTransformers(
#     model = 'llama-2-7b-chat.ggmlv3.q3_K_L.bin',
#     model_type = 'llama'
# )



# retriever_from_llm = MultiQueryRetriever.from_llm(
#     retriever=db.as_retriever(), llm=llm_llama2
# )

# # 관련된 문서들을 가지고 와서
# docs = retriever_from_llm.get_relevant_documents(query=question)
# print(len(docs))
# print(docs)
