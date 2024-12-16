import multiprocessing.queues
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import os
import config
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

import re
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Optional, Union, List, Dict, Tuple, Iterable, Callable, Any
from pdf_parser import PDFParser
from loguru import logger
import pickle
import numpy as np
import ray
from utils import save_pkl, load_pkl


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """Recursive text splitter for Chinese text.
    copy from: https://github.com/chatchat-space/Langchain-Chatchat/tree/master
    """

    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]
        self._is_separator_regex = is_separator_regex

    @staticmethod
    def _split_text_with_regex_from_end(
            text: str, separator: str, keep_separator: bool
    ) -> List[str]:
        # Now that we have the separator, split the text
        if separator:
            if keep_separator:
                # The parentheses in the pattern keep the delimiters in the result.
                _splits = re.split(f"({separator})", text)
                splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
                if len(_splits) % 2 == 1:
                    splits += _splits[-1:]
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = self._split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip() != ""]
    
    

@ray.remote(num_gpus=1)
def my_from_documents( docs):
    embedding = HuggingFaceEmbeddings(model_name=config.EMBEDDING_NAME,
                                            cache_folder =config.EMBEDDING_CACHE_FOLDER,
                                            model_kwargs={'device': "cuda"})
    db = FAISS.from_documents(docs, embedding)
    return db


import multiprocessing

def load_document_contents(filepath: str):
    _, file_extension = os.path.splitext(os.path.basename(filepath))
    try:
        if file_extension == ".pdf":
            with PDFParser(filepath) as parser:
                content, page_content_dict = parser.parse_all_pages()
                pdf_content = {os.path.basename(filepath): page_content_dict}
            return [content], pdf_content, None
        elif file_extension in [".docx", ".pptx", ".epub", '.doc']:
            loader_class = {
                ".docx": UnstructuredWordDocumentLoader,
                ".doc": UnstructuredWordDocumentLoader,
                ".pptx": UnstructuredPowerPointLoader,
                ".epub": UnstructuredEPubLoader
            }.get(file_extension)
            if loader_class:
                loader = loader_class(filepath)
                return loader.load(), None, None
        else:
            loader = TextLoader(filepath, "utf8")
            return loader.load(), None, None
    except Exception as e:
        logger.error(f"Error loading document {filepath}: {e}")
        return None, None, filepath




def load_document_contents_with_timeout(filepath, timeout):
    def target(queue, filepath):
        text, meta, errorF = load_document_contents(filepath)
        queue.put((text, meta, errorF))
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(queue, filepath))
    process.start() 
    
    try:
        text, meta, errorF = queue.get(timeout=timeout)
        process.join()
        return text, meta, errorF
    except multiprocessing.queues.Empty:
        logger.error(f"Timeout loading document {filepath}")
        process.terminate()
        process.join()
        return None, None, filepath

@ray.remote(num_cpus=1)
def ray_scan(filepaths):
    texts = []
    pdf_contents = {}
    error_files = []
    for filepath in filepaths:
        logger.info(f"Loading document {filepath}")
        text, pdf_content, error_file = load_document_contents_with_timeout(filepath, config.TIMEOUT)
        if error_file is not None:
            error_files.append(error_file)
        if text is not None:
            texts.append(text)
        if pdf_content is not None:
            pdf_contents.update(pdf_content)

    return texts, pdf_contents, error_files


class PlanEmb:
    
    def __init__(self) -> None:
        self.embedding = HuggingFaceEmbeddings(model_name=config.EMBEDDING_NAME,
                            cache_folder =config.EMBEDDING_CACHE_FOLDER,
                            model_kwargs={'device': "cuda"})
        self.text_splitter = ChineseRecursiveTextSplitter(
            chunk_size=500,
            chunk_overlap=0,
            length_function=len
        )
        self.db = None
        self.pdf_page2content_dic = {}
        self.index_path = None
        self.last_image = None
        self.filenames = []
        
        
    def get_all_file_paths(self, input_path):
        file_paths = [] 
        if os.path.isfile(input_path):
            file_paths.append(input_path)
        elif os.path.isdir(input_path):
            for root, dirs, files in os.walk(input_path):
                for file in files:
                    file_paths.append(os.path.join(root, file))
        else:
            print("Invalid path")
        return file_paths
    
    
    def _ray_embedding(self, docs):
        # chunks = [docs[i:i + len(docs) // config.DEVICES_NUMBER ] for i in range(0, len(docs), len(docs) // config.DEVICES_NUMBER)]
        step_size = max(1, len(docs) // config.DEVICES_NUMBER)
        chunks = [docs[i:i + step_size] for i in range(0, len(docs), step_size)]

        for i in range(len(chunks)):
            print(f"Chunk {i} has {len(chunks[i])} examples")
        tasks = [my_from_documents.remote(chunk) for chunk in chunks]
        processed_data = ray.get(tasks)
        # logger.debug(f"Processed data: {processed_data}")
        self.db = processed_data[0]
        for i in range(1, len(processed_data)):
            self.db.merge_from(processed_data[i])
        pass
    
    
    
    def build(self, database_path: str, index_path: str):
        # get all file paths
        file_paths = self.get_all_file_paths(database_path)
        self.index_path = index_path
        pdf2content_path = os.path.join(index_path, "pdf2content.pkl")
        if not os.path.exists(index_path):
            docs = self._get_documents(file_paths)
            logger.info(f"Indexing {len(docs)} documents")
            self._ray_embedding(docs)
            os.makedirs(index_path)
            self.save(index_path)
            logger.info(f"pdf_page2content saved to {pdf2content_path}")
        else:
            self.load(index_path)
            logger.info(f"Index path {index_path} already exists, load from {index_path}")
            
        
        
            
        
    def batch_build(self, database_path: str, index_path: str, batch_size: int = 10000):
        file_paths = self.get_all_file_paths(database_path)
        self.index_path = index_path
        if not os.path.exists(index_path):
            docs = self._get_documents(file_paths)
            logger.info(f"Indexing {len(docs)} documents")
            # embedding every 10000 document 
            count = 1
            for i in range(0, len(docs), batch_size):
                chunk_path = f"{index_path}/{count}"
                self._ray_embedding(docs[i:i+batch_size])
                # self.db = FAISS.from_documents(self.embedding, docs)
                os.makedirs(chunk_path)
                self.save(chunk_path)
                count += 1
                self.db = None
                self.pdf_page2content_dic = {}
                self.filenames = []
                logger.info(f"Chunk {count} saved to {chunk_path}")
            # merge chunk
            self.load(f"{index_path}/1")
            for i in range(2, count):
                self.merge_from_index(f"{index_path}/{i}")
            self.save(index_path)
        else:
            self.load(index_path)
            logger.info(f"Index path {index_path} already exists, load from {index_path}")
            
            
    def save(self, index_path: str):
        if os.path.exists(os.path.join(index_path, "pdf2content.pkl")) or os.path.exists(os.path.join(index_path, "filenames.pkl")):
            logger.warning(f"pdf2content.pkl or filenames.pkl already exists in {index_path}")
        if self.db is not None:
            self.db.save_local(index_path)
            save_pkl(self.pdf_page2content_dic, os.path.join(index_path, "pdf2content.pkl"))
            save_pkl(self.filenames, f"{index_path}/filenames.pkl")
        else:
            logger.error("No db to save")
        pass
    

    def merge_index(self, index_path: str, new_index_path: str):
        if self.db is None:
            self.load(index_path)
            self.pdf_page2content_dic = load_pkl(os.path.join(index_path, "pdf2content.pkl"))
            self.filenames = load_pkl(f"{index_path}/filenames.pkl")
        else:
            db = FAISS.load_local(index_path, self.embedding, allow_dangerous_deserialization = True)
            pdf_page2content_dic = load_pkl(os.path.join(index_path, "pdf2content.pkl"))
            filenames = load_pkl(f"{index_path}/filenames.pkl")
            self.db.merge_from(db)
            self.pdf_page2content_dic.update(pdf_page2content_dic)
            self.filenames.extend(filenames)
            self.save(new_index_path)
        pass
        
    def merge_from_index(self, index_path: str):
        if self.db is None:
            logger.error("No db to merge")
        else:
            db = FAISS.load_local(index_path, self.embedding, allow_dangerous_deserialization = True)
            self.db.merge_from(db)
            filenames = load_pkl(f"{index_path}/filenames.pkl")
            self.filenames.extend(filenames)
            self.pdf_page2content_dic.update(load_pkl(os.path.join(index_path, "pdf2content.pkl")))
        pass
            
        
    def load(self, index_path: str):
        logger.info(f"Load index from {index_path}")
        self.db = FAISS.load_local(index_path, self.embedding, allow_dangerous_deserialization = True)
        self.pdf_page2content_dic = load_pkl(os.path.join(index_path, "pdf2content.pkl"))
        self.filenames = load_pkl(f"{index_path}/filenames.pkl")
        self.index_path = index_path
        pass


    def add_documents_and_save(self, file_path: str, index_path: str):
        self.load(index_path)
        docs = self._get_documents([file_path])
        self._ray_embedding(docs)
        db = FAISS.from_documents(docs, self.embedding)
        self.db.merge_from(db)
        # remove all the files in the index_path
        for root, dirs, files in os.walk(index_path):
            for file in files:
                os.remove(os.path.join(root, file))
        self.save(index_path)
        pass
        
    
    
    def _get_documents(self, file_paths: List[str]) -> List[object]:
        documents = []
        for filepath in file_paths:
            self.filenames.append(os.path.basename(filepath))
        step_size = max(1, len(file_paths) // config.CPU_NUMBER)
        chunks = [file_paths[i:i + step_size] for i in range(0, len(file_paths), step_size)]

        # chunks = [file_paths[i:i + len(file_paths) // config.CPU_NUMBER ] for i in range(0, len(file_paths), len(file_paths) // config.CPU_NUMBER)]
        tasks = [ray_scan.remote( chunk) for chunk in chunks]
        files = []
        error_files = []    
        for task in tasks:
            file, pdf_contents, error_file = ray.get(task)
            error_files.extend(error_file)
            files.extend(file)
            self.pdf_page2content_dic.update(pdf_contents)
        ## save error files
        if error_files:
            error_file_path = "error_files.txt"
            with open(error_file_path, "w") as f:
                for error_file in error_files:
                    f.write(f"{error_file}\n")
            logger.error(f"Error files saved to {error_file_path}")
        for file in files:
            if file:
                file = self.text_splitter.split_documents(file)
                documents.extend(file)
        return documents
        
    
    def get_topk(self, query: str, topk: int = 5):
        return self.db.similarity_search_with_score(query, topk)
            
            
            
if __name__ == '__main__':
    print("Start embedding")
    emb = PlanEmb()
    # emb.load('/home/admin/plangpt/plangpt-beijing/test_index/1')
    
    # emb.build(config.DATA_PATH, config.INDEX_PATH)
    report_path = './DATA/reports/raw/test.pdf'
    filename = os.path.basename(report_path)
    target_path = os.path.join(config.DB_PATH, filename.split('.')[0])
    print(target_path)
    emb.build(report_path, target_path)
    emb.load(target_path)
    tmp = emb.get_topk('论证重点是什么？', 2)
    print(tmp)



