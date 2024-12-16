
import os
from zipfile import ZipFile
import shutil
import pickle
from loguru import logger
# import nltk
# nltk.download('averaged_perceptron_tagger_eng')

pwd = os.path.dirname(__file__)
database_path = os.path.join(pwd, "database")

def save_file_or_folder(file, is_folder):
    if is_folder:
        with ZipFile(file.name, 'r') as zip_ref:
            zip_ref.extractall(database_path)
        os.remove(file.name)  
        return "文件夹上传成功！"
    else:
        print("file: ", file)
        save_path = os.path.join(database_path, file.name.split("/")[-1])
        print("save_path: ", save_path)
        shutil.copy(file.name, save_path)
        return "文件上传成功！"

from tqdm import tqdm
import json

def load_txt(filepath):
    # 首先，打开并读取文件内容
    with open(filepath, 'r', encoding='utf-8') as file:
        # 读取所有行到一个列表中
        lines = file.readlines()

    # 初始化一个空列表用于存放处理后的意见
    content = []

    # 循环遍历每一行进行处理
    for line in lines:
        if(line != '\n'):
            content.append(line)

    # 打印处理后的意见列表
    # print(content)
    return content


def list_files():
    files = os.listdir(database_path)
    return "\n".join(files)

def load_jsonlines(filepath):
    with open(filepath, "r") as f:
        return [json.loads(line) for line in tqdm(f)]

def save_jsonlines(data, filepath):
    if os.path.exists(filepath):
        print("File exists, please check the path.")
        return
    with open(filepath, "w") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(data, file_path):
    with open(file_path, 'wb') as f:
        logger.info(f"Saving data to {file_path}")
        pickle.dump(data, f)

def n_grams(text, n):
    text = text.replace(" ", "")
    return [text[i:i+n] for i in range(len(text)-n+1)]


from langchain_community.vectorstores import FAISS
# from langchain_openai.embeddings import OpenAIEmbeddings
import os


from PyPDF2 import PdfReader
import fitz  # 导入PyMuPDF库
import re

def save_pdf_img(path,save_path):
    '''
    path: pdf的路径
    save_path : 图片存储的路径
    '''
    # 使用正则表达式来查找图片
    checkXO = r"/Type(?= */XObject)" 
    checkIM = r"/Subtype(?= */Image)"  
    
    # 打开pdf
    doc = fitz.open(path)
    # 图片计数
    imgcount = 0
    # 获取对象数量长度
    lenXREF = doc.xref_length()
 
    # 打印PDF的信息
    print("文件名:{}, 页数: {}, 对象: {}".format(path, len(doc), lenXREF - 1))
    
    
    # 遍历每一个对象
    for i in range(1, lenXREF):
        # 定义对象字符串
        text = doc.xref_object(i)
#         print(i,text)

        isXObject = re.search(checkXO, text)
        # 使用正则表达式查看是否是图片
        isImage = re.search(checkIM, text)
        
        # 如果不是对象也不是图片，则continue
        if not isXObject or not isImage:
            continue
        imgcount += 1
        # 根据索引生成图像
        pix = fitz.Pixmap(doc, i)
        # 根据pdf的路径生成图片的名称
        new_name = path.replace('\\', '_') + "_img{}.png".format(imgcount)
        new_name = new_name.replace(':', '')
        # 如果pix.n<5,可以直接存为PNG
        if pix.n < 5:
            pix.writePNG(os.path.join(save_path, new_name))
        # 否则先转换CMYK
        else:
            pix0 = fitz.Pixmap(fitz.csRGB, pix)
            pix0.writePNG(os.path.join(save_path, new_name))
            pix0 = None
        # 释放资源
        pix = None
        print("提取了{}张图片".format(imgcount))

class pdf_process:

    def __init__(self, pdf):
        self.pdf = pdf

    # 获取pdf文件内容
    def get_pdf_text(self):
        text = ""
        pdf_reader = PdfReader(self.pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

        return text

from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader

#加载md文件
def load_md_file(md_file):    
    loader = UnstructuredMarkdownLoader(md_file)
    docs = loader.load()
    print(docs[0].page_content[:100])
    return docs


#分割md文件
def load_md_splitter(md_file, chunk_size=200, chunk_overlap=20):
    docs = load_md_file(md_file)
    text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = text_splitter.split_documents(docs)
    #默认展示分割后第一段内容
    print('split_docs[0]: ', split_docs[0])
    return split_docs

# 读取Markdown文件
def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

# 拆分文本
def get_text_chunks(
        text,
        chunk_size: int = 768,
        chunk_overlap: int = 200
    ):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        # chunk_size=768,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(text)
