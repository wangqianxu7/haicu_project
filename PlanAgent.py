        
from typing import Dict, Union, List
from config import *
from loguru import logger
from PlanRAG import PlanRAG
from client import ChatQwen
import re
import markdown
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import os

os.environ['CUDA_VISIBLE_DEVICES']="4"

def clean_qwen2(response: str):
    response = response.rstrip()
    response = response.rstrip('\n')
    if response.endswith("'"):
        response = response[:-1]
        return clean_qwen2(response)
    elif response.endswith("')"):
        response = response[:-2]
        return clean_qwen2(response)
    elif response.endswith("'("):
        response = response[:-2]
        return clean_qwen2(response)
    elif response.endswith(")'"):
        response = response[:-2]
        return clean_qwen2(response)
    elif response.endswith("('"):
        response = response[:-2]
        return clean_qwen2(response)
    elif response.endswith(")"):
        response = response[:-1]
        return clean_qwen2(response)
    elif response.endswith("("):
        response = response[:-1]
        return clean_qwen2(response)
    else:
        return response
    return response

class PlanAgent:

    def __init__(self, plan_rag = None, client = None):
        self.plan_rag = plan_rag
        self.client = client


    def get_answer_from_db(
            self,
            query : Union[str],
            top_k : Union[int],
            fetch_n: Union[int]) -> Dict[str,str]:

        if top_k == 0:
            return {"input": "根据已知信息无法回答该问题,正在启动网络检索",
                    "info": "根据已知信息无法回答该问题,正在启动网络检索",
                    "db_ref": "没有找到相关引用",
                    "image": ""}
        else:
            # concat_context =''.join([files[i]['context'] for i in range(top_k)])
            files,image =  self.plan_rag.advance_search(query,  fetch_n)
            concat_context = ''
            for i in range(top_k):
                # files[i]['context'] = files[i]['context'].replace('\n', '')
                concat_context += f"【信息{i}】来自于{os.path.basename(files[i]['metadata'])}\n:{files[i]['context']}\n"
            # if self.check_related(query, concat_context):
            #     DB_PROMPT_TEMPLATE_FINISH = DB_PROMPT_TEMPLATE.format(context = concat_context,question=query)
            # else:
            #     DB_PROMPT_TEMPLATE_FINISH = PLANGPT_TEMPLATE.format(question=query)
            # DB_INTERGRATE_TEMPLATE_FINISH = DB_INTERGRATE_TEMPLATE.format(context = concat_context)
            # concat_context, _ = self.client.chat(DB_INTERGRATE_TEMPLATE_FINISH, history=[])
            concat_context = re.sub(r'\n\n(?=[2-9]|[1-9]\d)', '\n', concat_context)
            # concat_context = clean_qwen2(concat_context)
            DB_PROMPT_TEMPLATE_FINISH = DB_PROMPT_TEMPLATE.format(context = concat_context,question=query)
            db_ref = [{'source': item['metadata'], "text": self.clean_markdown(item['context']) } for id, item in enumerate(files)]
            # logger.debug(f"db_ref: {db_ref}")
            return {"input": DB_PROMPT_TEMPLATE_FINISH,
                    "info": concat_context,
                    "db_ref": db_ref,
                    "image": image}
        
    def check_retrieval(self, query: str, history: List[Dict[str,str]]):
        _, __  = self.client.chat(RETRIEVAL_TEMPLATE.format(query=query),history=history)
        _ = clean_qwen2(_)
        logger.info(f"是否需要检索: {_}")
        if "0" in _:
            logger.info("不需要检索数据库")
            return False
        return True

    def format_history(self, history):
        formatted_history = []
        if len(history)%2 == 0:
            for i in range(0, len(history), 2):
                formatted_history.append((history[i]['message'], history[i+1]['message']))
            return formatted_history
        return []

    def clean_markdown(self, text: str):
        html = markdown.markdown(text)
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()
        text = text.replace('\n', '')
        return text

    def chat_LLM(self, query: str, topk: int, history: List, fetch_n: int = 5):
        # only for our front-end
        # history = self.format_history(history)
        logger.info(f"history: {history}")
        if not self.check_retrieval(query, history):
            topk = 0
        db_response = self.get_answer_from_db(query, topk, fetch_n=topk)
        input_text = db_response['input'] if "没有找到相关引用" not in db_response['db_ref'] else PLANGPT_TEMPLATE.format(question=query)
        logger.debug(f"input_text: {input_text}")

        # info_tag = "根据引用回答" if "没有找到相关引用" not in db_response['db_ref'] else "使用模型自己的能力回答问题"
        llm_response, _ = self.client.chat(input_text , history=history)
        llm_response = re.sub(r'\n\n(?=[2-9]|[1-9]\d)', '\n', llm_response)
        llm_response = clean_qwen2(llm_response)
        if "根据已知信息无法回答该问题" in llm_response:
            input_text = PLANGPT_TEMPLATE.format(question=query)
            logger.debug(f"input_text: {input_text}")
            llm_response, _ = self.client.chat(input_text , history=history)
            llm_response = re.sub(r'\n\n(?=[2-9]|[1-9]\d)', '\n', llm_response)
            llm_response = clean_qwen2(llm_response)
            topk = 0
        # history.append((query, llm_response))
        history.append({"role":"user", "message": query})
        history.append({"role":"LLM", "message": llm_response})
        logger.debug(f"llm_response: {llm_response}")
        logger.debug(f"history: {history}")
        if topk == 0:
            db_ref_text = "不需要检索数据库"
        else:
            db_ref_text = db_response['db_ref']
            db_ref_text = [item['text'] for item in db_ref_text]
            db_ref_text = db_ref_text[0]
        return llm_response, history, db_ref_text
    
class PlanCheckAgent:
    
    def __init__(self, plan_rag = None, client = None):
        self.client = client
        self.plan_rag = plan_rag
    
    def check_page(self, page_content):
        response, __  = self.client.chat(CHECKDOC_PROMPT_TEMPLATE.format(content=page_content))
        # logger.info(f"response_page:{response}")
        return response
    
    def check_laws(self):

        query = "请你找出该论证报告中所有的法律、法规、规章等材料。"

        laws_content, _ = self.plan_rag.advance_search(query, 6)
        # logger.info(f"laws_content:{laws_content}")
        response, __  = self.client.chat(CHECKLAWS_PROMPT_TEMPLATE.format(content=laws_content))
        response = re.sub(r'\n\n(?=[2-9]|[1-9]\d)', '\n', response)
        response = clean_qwen2(response)
        logger.info(f"response_laws:{response}")
        return response
    
    def integration_reports(self):
        text = ""
        pdf_reader = PdfReader(self.report_path)
        for page in pdf_reader.pages:
            content = page.extract_text()
            text += self.check_page(content)

        text += self.check_laws()
        
        return text

if __name__ == "__main__":
    client = ChatQwen()
    plan_rag = PlanRAG(INDEX_PATH)
    agent = PlanAgent(plan_rag, client)
    # test_query = "海域使用类型为油气开采用海，其论证重点为什么？"
    test_query = "强化学习是什么？"
    test_history = []
    response, history_rs, db_ref = agent.chat_LLM(test_query, 1, test_history)
    logger.info(f"response: {response}")
    logger.info(f"history: {history_rs}")
    logger.info(f"db_ref: {db_ref}")