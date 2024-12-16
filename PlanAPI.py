from fastapi import FastAPI, UploadFile, File, Form
import os
import shutil
import re
import time
from loguru import logger
import uvicorn


from PlanCheck import extract_content, extract_sentences, clean_qwen2
from PlanAudit import generate_doc
from PlanAgent import PlanAgent
from PlanEmb import PlanEmb
from PlanRAG import PlanRAG
from client import ChatQwen
from config import *

# 初始化
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
client = ChatQwen()
emb = PlanEmb()
plan_rag = PlanRAG(INDEX_PATH)
plan_agent = PlanAgent(plan_rag, client)

# FastAPI 应用实例
app = FastAPI(
    title="PlanGPT",
    description="基于RAG的智能审查与问答服务",
    version="1.0.0",
)

# 创建临时文件夹
os.makedirs("tmp", exist_ok=True)
os.makedirs("output", exist_ok=True)


# 工具函数：格式检查逻辑
def format_check_logic(format_type: str, filepath: str) -> str:
    blank_pages, chapters, titles, attachments, images, tables, m_images, m_tables = extract_content(filepath)
    history = []
    response = ""

    if "空白" in format_type:
        if blank_pages == []:
            response = "空白页检查：\n暂未发现该论证报告含有空白页。\n"
        else:
            response = "空白页检查：\n该论证报告存在空白页，如下所示："+blank_pages+"。"

    elif "章节" in format_type:
        CHAPTER_PROMPT_TEMPLATE_FINISH = CHAPTER_PROMPT_TEMPLATE.format(content=chapters)

        llm_response, history = client.chat(CHAPTER_PROMPT_TEMPLATE_FINISH , history=history)
        llm_response = re.sub(r'\n\n(?=[2-9]|[1-9]\d)', '\n', llm_response)
        llm_response = clean_qwen2(llm_response)
        response = "章节错乱检测（出现漏节、跳节）：\n"+llm_response+"\n"

    elif "标题" in format_type:
        TITLE_PROMPT_TEMPLATE_FINISH = TITLE_PROMPT_TEMPLATE.format(content=titles)

        llm_response, history = client.chat(TITLE_PROMPT_TEMPLATE_FINISH , history=history)
        llm_response = re.sub(r'\n\n(?=[2-9]|[1-9]\d)', '\n', llm_response)
        llm_response = clean_qwen2(llm_response)
        response = "标题目录级别弄混、重复：\n"+llm_response+"\n"

    elif "错别字" in format_type:
        sentences = extract_sentences(filepath)
        count = 0
        for sentence in sentences[1:20]:
            CUOBIEZI_PROMPT_TEMPLATE_FINISH = CUOBIEZI_PROMPT_TEMPLATE.format(content=sentence)
            llm_response, history = client.chat(CUOBIEZI_PROMPT_TEMPLATE_FINISH , history=history)
            llm_response = re.sub(r'\n\n(?=[2-9]|[1-9]\d)', '\n', llm_response)
            llm_response = clean_qwen2(llm_response)
            if "明显" not in llm_response:
                response += llm_response + "\n"
                count += 1 # 计数
        
        if count != 0:
            response = "错别字、书写有误(部分)： \n"+response+"\n"
        else:
            response = "错别字、书写有误(部分)： \n"+ "暂未发现有错别字、书写有误" +"\n"

    elif "报告" in format_type:
        ATTACHMENT_PROMPT_TEMPLATE_FINISH = ATTACHMENT_PROMPT_TEMPLATE.format(content=attachments)
    
        llm_response, history = client.chat(ATTACHMENT_PROMPT_TEMPLATE_FINISH , history=history)
        llm_response = re.sub(r'\n\n(?=[2-9]|[1-9]\d)', '\n', llm_response)
        llm_response = clean_qwen2(llm_response)
        response = "报告固定内容缺失：\n"+llm_response+"\n"

    elif "序号" in format_type:
        TUBIAO_PROMPT_TEMPLATE_FINISH = TUBIAO_PROMPT_TEMPLATE.format(image_content=images,table_content=tables)

        llm_response, history = client.chat(TUBIAO_PROMPT_TEMPLATE_FINISH , history=history)
        llm_response = re.sub(r'\n\n(?=[2-9]|[1-9]\d)', '\n', llm_response)
        llm_response = clean_qwen2(llm_response)
        response = "图表序号错乱、重复、遗漏等: \n" +llm_response + "\n"

    elif "索引" in format_type:
        TUBIAOCHECK_PROMPT_TEMPLATE_FINISH = TUBIAOCHECK_PROMPT_TEMPLATE.format(image_mention=m_images,image_content=images,table_mention=m_tables,table_content=tables)

        llm_response, history = client.chat(TUBIAOCHECK_PROMPT_TEMPLATE_FINISH , history=history)
        llm_response = re.sub(r'\n\n(?=[2-9]|[1-9]\d)', '\n', llm_response)
        llm_response = clean_qwen2(llm_response)
        response = "报告内容漏写图表号、报告中提及图表号但无对应图表: \n" +llm_response + "\n"

    return response


# 工具函数：审查逻辑
def get_audit_report(filepath: str, audit_re: str) -> str:
    input_path = f"tmp/{os.path.basename(filepath).split('.')[0]}"
    if not os.path.exists(input_path):
        emb.build(filepath, input_path)

    history = []
    audit_agent = PlanAgent(PlanRAG(input_path), client)
    response, _ = audit_agent.chat_LLM(audit_re, "审查依据", 1, 3, history)
    return response


# 工具函数：生成综合报告
def generate_audit_doc(filepath: str) -> str:
    filename = os.path.basename(filepath).split('.')[0]
    target_dir = f"output/{filename}"
    os.makedirs(target_dir, exist_ok=True)
    output_path = os.path.join(target_dir, "AI智能审查报告.docx")
    generate_doc(filepath, output_path)
    return output_path


# 路由：格式检查
@app.post("/format/check")
async def format_check(
    format_type: str = Form(...),
    file: UploadFile = File(...)
):
    file_path = f"tmp/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    response = format_check_logic(format_type, file_path)
    return {"status": "success", "data": response}


# 路由：AI 审查
@app.post("/audit/generate_report")
async def audit(
    audit_re: str = Form(...),
    file: UploadFile = File(...)
):
    file_path = f"tmp/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    report = get_audit_report(file_path, audit_re)
    return {"status": "success", "report": report}


@app.post("/audit/generate_doc")
async def generate_doc_route(file: UploadFile = File(...)):
    file_path = f"tmp/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    doc_path = generate_audit_doc(file_path)
    return {"status": "success", "doc_path": doc_path}


# 路由：知识库问答
@app.post("/chatbot/ask")
async def chatbot_ask(
    question: str = Form(...)
):
    response, _, sources = plan_agent.chat_LLM(question, 1, [])
    return {"status": "success", "response": response, "sources": sources}


# 启动检查
@app.get("/")
def read_root():
    return {"message": "Welcome to PlanGPT API"}

if __name__=='__main__':
    uvicorn.run(app,
                host=None,
                port=23454,
                log_level="debug")
