from typing import List, Dict
from utils import save_pkl, load_pkl
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
import fitz
import numpy as np
from PIL import Image
from fuzzywuzzy import fuzz
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain_community.vectorstores import FAISS
import ray
import os
from zhkeybert import KeyBERT, extract_kws_zh
from sentence_transformers import SentenceTransformer
import config  # 需要导入config模块以访问EMBEDDING_NAME和EMBEDDING_CACHE_FOLDER
import time

os.environ['CUDA_VISIBLE_DEVICES']="4"

class PlanRAG:
    '''
    PlanRAG is a class that can be used to search for relevant documents in a database.
    '''
    def __init__(self, index_path: str) -> None:
        self.embedding = HuggingFaceEmbeddings(model_name=config.EMBEDDING_NAME,
                                                cache_folder =config.EMBEDDING_CACHE_FOLDER,
                                                model_kwargs={'device': "cuda"})
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(config.RERANK_MODEL_PATH)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(config.RERANK_MODEL_PATH).cuda()
        self.rerank_model.eval()
        self.doc_size = 0
        self.db = FAISS.load_local(index_path, self.embedding, allow_dangerous_deserialization=True)
        # 现在是{fileName: {page_num: content}}
        self.pdf_page2content_dic = load_pkl(os.path.join(index_path, "pdf2content.pkl"))
        self.filenames = load_pkl(f"{index_path}/filenames.pkl")
        self.index_path = index_path
        self.last_image = None
        sentence_model = SentenceTransformer(config.EMBEDDING_NAME)
        self.kw_model = KeyBERT(model=sentence_model)
        
    
    def __repr__(self) -> str:
        return "<PlanRAG with {} documents>".format(len(self.db))

    def __len__(self) -> int:
        return len(self.db)

    def __getitem__(self, index: int):
        return self.db[index]
    
    
    def render_file(self, page_number, source):
        doc = fitz.open(os.path.join(self.db_path, source))
        page = doc[page_number]
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.frombytes('RGB', [pix.width, pix.height], pix.samples)
        return image
    
    def find_page_num(self, content: str, source: str):

        # #对content 用\n 分割
        # content = content.split("\n")
        # # 过滤掉长度<10的
        # content = [c for c in content if len(c) > 10]
        if not self.pdf_page2content_dic:
            # logger.info(f"No pdf page content found.")
            return 0
        results = []
        content_tmp = content
        for page_num, page_content in self.pdf_page2content_dic[source].items():
            if len(content_tmp) <= len(page_content):
                results.append((page_num, fuzz.partial_ratio(content_tmp, page_content))) 
        results = sorted(results, key=lambda x: x[1], reverse=True)
        # logger.debug(f"Results: {results[:10]}")
        # logger.info(f"Founded Page number: {results[0][0]}")
        page_nums = [page_num for page_num, score in results[:5]]
        # return page_nums
        return results[0][0]
        # logger.info(f"Page number not found")
        # return 0
        

    def exact_search(
        self, query: str, top_k: int
    ) -> List[Dict[str, str]]:
        results_with_scores = self.db.similarity_search_with_score(query, k=top_k)
        return [
            {
                'content': doc.page_content,
                'source': os.path.basename(doc.metadata['source']),
                'context': "",
                'metadata': doc.metadata['source'].split('/')[-1],
                'score': score
            } for doc, score in results_with_scores
        ]

    

    def embed_query(self, query):
        embedding = self.embedding.embed_query(query)
        return np.array([embedding], dtype=np.float32)

    def fetch_document(self, doc_id):
        """Fetches a document and its metadata by document ID."""
        doc = self.db.docstore.search(doc_id)
        return doc.page_content, doc.metadata['source']

    def get_context(self, doc_id, target_ref, max_context_number):
        """Gathers context around a given document ID within the same source document."""
        contents = []
        # 前2个后2个，注意越界
        for i in range(-max_context_number, max_context_number + 1):
            cur_id = doc_id + i
            if not (0 <= cur_id < len(self.db.index_to_docstore_id)):
                continue
            cur_doc_id = self.db.index_to_docstore_id[cur_id]
            cur_content, cur_source = self.fetch_document(cur_doc_id)
            if cur_source.split('/')[-1] == target_ref:
                contents.append(cur_content)
        return ''.join(contents)
    

    def context_search(self, query, top_f=30, max_context_number=1):
        vector = self.embed_query(query)
        scores, indices = self.db.index.search(vector, top_f)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            doc_id = int(idx)
            doc_content, doc_source = self.fetch_document(self.db.index_to_docstore_id[doc_id])
            target_ref = doc_source.split('/')[-1]
            context = self.get_context(doc_id, target_ref, max_context_number)
            if len(context) < 50:
                continue
            results.append({
                'content': doc_content,
                'metadata': doc_source,
                'context': context,
                'score': score,
                'id': doc_id,
                'tscore': 1, # 这个tscore
            })
        results.sort(key=lambda x: (-x['tscore'], x['score']))
    
        return results
    
    def filter_search(self, query, fileName, top_k=5):
        results_with_scores = self.db.similarity_search_with_score(query, k=top_k, filter=dict(source=fileName))
        return [
            {
                'content': doc.page_content,
                'source': os.path.basename(doc.metadata['source']),
                'context': "",
                'metadata': doc.metadata['source'].split('/')[-1],
                'score': score
            } for doc, score in results_with_scores
        ]

    
 
    def advance_search(self, query, top_k=3):
        '''
        return {content, metadata, context, score, id, tscore}
        '''
        if self.db is None or top_k == 0:
            if self.last_image is not None:
                return [], self.last_image
            return [], None
        # logger.info(f"Top_k: {top_k}")
        results = self.context_search(query) # default is 10  # 
        final_results = results
        if len(final_results) < top_k:
            top_k = len(final_results)
        # final_results = self.keyword_reranking(final_results, query)[:30]
        final_results = self.model_reranking(final_results, query)[:10]
        image = None
        for item in final_results:
            item["page_num"] = self.find_page_num(item["content"], os.path.basename(item["metadata"]))
        return final_results[:top_k], image
    
    def _extract_keywords(self, query):
        return extract_kws_zh(query, self.kw_model, ngram_range=(1, 2), top_n=20)
        
    def keyword_reranking(self, results, query):
        keywords = self._extract_keywords(query)
        for result in results:
            context_keywords = self._extract_keywords(result['context'])
            score = 0
            for keyword in keywords:
                for context_keyword in context_keywords:
                    score += fuzz.partial_ratio(keyword, context_keyword)
            result['keyword_score'] = score
        results = sorted(results, key=lambda x: x['keyword_score'], reverse=True)
        return results
    
    
    def model_reranking(self, results, query):
        pairs = [[query, result['content']] for result in results]
        self.rerank_tokenizer
        with torch.no_grad():
            inputs = self.rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512) #TODO: 512 作为self.rerank_model_dim 参数
            inputs = {k: v.cuda() for k, v in inputs.items()}
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
            # logger.debug(f"Rerank scores: {scores}")
        for result, score in zip(results, scores):
            result['rerank_score'] = score
        results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        return results
    
if __name__ == "__main__":
    time_start = time.time()
    pr = PlanRAG(config.INDEX_PATH)
    # query = "海域使用类型为开放式养殖用海，如筏式养殖、网箱养殖及无人工设施的人工投苗或自然增殖生产等的用海，其论证重点包括哪些部分？"
    query = "海域使用类型为油气开采用海，其论证重点为什么？"
    # data =pr.pg_search(query, 3)
    data, _ = pr.advance_search(query, 3)
    print(data)
    concat_context = ''
    for i in range(3):
        concat_context += f"【信息{i}】来自于{os.path.basename(data[i]['metadata'])}\n:{data[i]['context']}\n页码信息:{data[i]['page_num']}\n"

    print(concat_context)
    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)
