import os
import re

import gradio as gr  # 导入Gradio库
import time  # 导入time库，用于控制延迟
# from PlanAgent import test
from PlanCheck import extract_content, extract_sentences, mkdoc, clean_qwen2
from utils import load_txt
from PlanRAG import PlanRAG
from client import ChatQwen
from PlanEmb import PlanEmb
from PlanAgent import PlanAgent
import shutil
import logging
from config import *

from PlanAudit import PlanAudit, generate_doc
from loguru import logger

os.environ['CUDA_VISIBLE_DEVICES']="4"

logger = logging.getLogger(__name__)
client = ChatQwen()
emb = PlanEmb()
plan_rag = PlanRAG(INDEX_PATH)
plan_agent = PlanAgent(plan_rag, client)

def format_check(format_type, filepath):
    blank_pages, chapters, titles, attachments, images, tables, m_images, m_tables = extract_content(filepath)

    history = []
    response = ""
    if "空白" in format_type:
        logger.info(f"blank_pages:{blank_pages}")
        if blank_pages == []:
            response = "空白页检查：\n暂未发现该论证报告含有空白页。\n"
        else:
            response = "空白页检查：\n该论证报告存在空白页，如下所示："+ ", ".join(map(str, blank_pages))+"。"

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
        logger.info(f"images:{images}, tables:{tables}")
        llm_response, history = client.chat(TUBIAO_PROMPT_TEMPLATE_FINISH , history=history)
        llm_response = re.sub(r'\n\n(?=[2-9]|[1-9]\d)', '\n', llm_response)
        llm_response = clean_qwen2(llm_response)
        response = "图表序号错乱、重复、遗漏等: \n" +llm_response + "\n"
        logger.info(f"llm_response_图表序号问题:{response}")

    elif "索引" in format_type:
        TUBIAOCHECK_PROMPT_TEMPLATE_FINISH = TUBIAOCHECK_PROMPT_TEMPLATE.format(image_mention=m_images,image_content=images,table_mention=m_tables,table_content=tables)

        llm_response, history = client.chat(TUBIAOCHECK_PROMPT_TEMPLATE_FINISH , history=history)
        llm_response = re.sub(r'\n\n(?=[2-9]|[1-9]\d)', '\n', llm_response)
        llm_response = clean_qwen2(llm_response)
        response = "报告内容漏写图表号、报告中提及图表号但无对应图表: \n" +llm_response + "\n"

    return response

def obtain_fund(audit_re):
    filepath = "./DATA/guideline/audit_re.txt"
    filepath2 = "./DATA/guideline/audit_foundation.txt"
    audit_sug = load_txt(filepath)
    audit_foundation = load_txt(filepath2)
    for i in range(len(audit_sug)):
        if audit_re in audit_sug[i]:
            return audit_foundation[i]
    return None

def save_file(file_obj, target_dir):
    """
    保存上传的文件到目标目录，避免重复保存。
    :param file_obj: 上传的文件对象
    :param target_dir: 目标目录
    :return: 保存后的文件路径
    """
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, os.path.basename(file_obj.name))
    
    if not os.path.exists(target_path):
        shutil.copy(file_obj.name, target_path)
        logger.info(f"文件已保存到: {target_path}")
    else:
        logger.info(f"文件已存在，跳过保存: {target_path}")
    
    return target_path

def process_file(file_path, process_function):
    """
    通用的文件处理逻辑。
    :param file_path: 输入文件路径
    :param process_function: 文件处理函数（如 mkdoc 或 generate_doc）
    :param output_name: 输出文件名称
    :return: 生成的输出文件路径
    """
    filename = os.path.basename(file_path)
    input_dir = os.path.join(DB_PATH, filename.split('.')[0])
    os.makedirs(input_dir, exist_ok=True)
    # 生成输出文件路径
    target_dir = os.path.join(OUT_PATH, filename.split('.')[0])
    os.makedirs(target_dir, exist_ok=True)
    
    # 如果处理函数是 mkdoc，使用 save_path 而不是 input_dir
    if process_function == mkdoc:
        output_path = process_function(file_path, filename)  # 传递 save_path 给 mkdoc
    else:
        # 否则，正常传递 input_dir 给 process_function
        output_path = process_function(filename)
    
    # output_path = os.path.join(target_dir, output_name)
    
    logger.info(f"生成的文件路径: {output_path}")
    return output_path

def generate_info(file_obj):
    saved_path = save_file(file_obj, DBR_PATH)
    input_path = os.path.join(DB_PATH, os.path.basename(saved_path).split('.')[0])
    
    if not os.path.exists(input_path):
        emb.build(saved_path, input_path)
    plan_rag = PlanRAG(input_path)
    audit_agent = PlanAudit(plan_rag, client)
    return audit_agent.base_info

def get_audit(file_obj, audit_re, base_info):
    """
    审查函数。
    """
    if not audit_re.strip():
        return "未输入审查要求，请提供明确的审查内容。"
    saved_path = save_file(file_obj, DBR_PATH)
    input_path = os.path.join(DB_PATH, os.path.basename(saved_path).split('.')[0])
    
    if not os.path.exists(input_path):
        emb.build(saved_path, input_path)

    history = []
    plan_rag = PlanRAG(input_path)
    audit_agent = PlanAudit(plan_rag, client, base_info)
    audit_fund = obtain_fund(audit_re)
    if audit_fund is None:
        return "对应审查依据未知，暂时无法根据此审查要求进行AI审查！"
    
    response, history_rs = audit_agent.chat_llm(audit_re, audit_fund, 1, 3, history)
    logger.info(f"response: {response}")
    return response

def generate_file1(file_obj):
    """
    生成格式审查综合报告。
    """
    saved_path = save_file(file_obj, DBR_PATH)
    shutil.copy(saved_path, 'tmpdir')  # 复制到临时目录

    return process_file(
        file_path=saved_path,
        process_function=mkdoc
    )

def generate_file2(file_obj):
    """
    生成AI智能审查报告。
    """
    saved_path = save_file(file_obj, DBR_PATH)
    shutil.copy(saved_path, 'tmpdir')  # 复制到临时目录

    return process_file(
        file_path=saved_path,
        process_function=generate_doc
    )

def clear():
    return "", "", gr.update(interactive=True), gr.update(interactive=True), None

def clear_all():
    return None, "", "", "", None

def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "\n\n"+line
    text = "".join(lines)
    return text

def predict(query, chatbot):
    global client, plan_rag
    parsed_query = parse_text(query)
    chatbot.append((parsed_query, ""))
    bot_message, history_rs, rag_docs = plan_agent.chat_LLM(parsed_query, 1, [])
    chatbot[-1] = (parsed_query, bot_message)
    logger.info(f"rag_docs:{rag_docs}")
    if "不需要" not in rag_docs and "无法回答该问题" not in bot_message:
        # 更新 db_resp 和 db_ref
        return chatbot, None, gr.update(value="海域使用论证导则.pdf"), gr.update(value=rag_docs)  # 清空 db_ref
    return chatbot, None, gr.update(value="根据模型本身能力回答"), gr.update(value="")

# 使用Gradio的Blocks构建一个应用程序界面
with gr.Blocks(
    title= "海域使用论证报告审查AI助手",
    theme = "Soft",
    css="style.css"
) as demo:
    gr.HTML("""<style>
                body { margin: 0; font-family: Arial, sans-serif; }
                .header { padding: 0px; text-align: center; color: white; font-size: 40px; font-weight: bold; border-radius:30px; }
                .adjustable-column { padding: 20px; }
                h5 { margin-top: 0px; position: relative; top: -10px; text-align: center; } /* 使用 position 和 top 属性进行调整 */
            </style>
            <div class="header">🏙️ 海域使用论证报告审查AI助手</div>
            """)

    with gr.Tab(label="知识问答"):
        title = gr.Markdown("# 政策问答大模型服务")
        with gr.Column():
            with gr.Row():
                chatbot = gr.Chatbot(value=[], elem_id='chatbot', height=570)  # 创建一个聊天机器人组件

                with gr.Column():
                    db_resp = gr.Textbox(label="信息来源", lines=2, max_lines=2, interactive=False)
                    db_ref = gr.Textbox(label="本地引用", lines=20, max_lines=20, interactive=False)

        with gr.Row():
            with gr.Column(scale=3):
                msg = gr.Textbox(show_label=False,
                                placeholder="请在这里输入你想提问的问题",
                                container=False)  # 创建一个文本框组件，用于用户输入消息
                
            with gr.Column(scale=1):
                submit_button = gr.Button('发送')
        
            with gr.Column(scale=1):
                clear_btn = gr.Button("清空")  # 创建一个按钮组件，用于清除聊天历史

        submit_button.click(predict, inputs=[msg, chatbot],
            outputs=[chatbot, msg, db_resp, db_ref], show_progress=True)
        # 当点击清除按钮时，清除聊天历史
        clear_btn.click(
            lambda: [[], None, None],  # Resets all outputs to their cleared or default state
        outputs=[chatbot, db_resp, db_ref],
        show_progress=True)
    
    with gr.Tab(label="格式审查"):
        title = gr.Markdown("# 智能化合规性检查服务")
        with gr.Column():
            file_input1 = gr.File(label="上传文件")
            choices = gr.Dropdown(
                ["空白页问题检测", "章节错乱问题检测", "标题错乱问题检测", "错别字问题检测","报告内容缺失检测",
                 "图表序号错乱检测","图表索引缺失检测"], label="类型选择", info="请选择需要审查的格式类型")
            with gr.Row():
                button = gr.Button("生成报告")
                button_c = gr.Button("清空")

            text_check = gr.Textbox(label="检测报告",interactive=False)
            button_w1 = gr.Button("生成综合报告")
            file_output1 = gr.File(label="下载文件")

        button.click(
            fn=lambda choices, file_input1: (
                "未选择检测类型，请选择需要审查的格式类型！" if not choices else
                "未上传文件，请上传所需的文件！" if not file_input1 else
                format_check(choices, file_input1)
            ),
            inputs=[choices, file_input1],
            outputs=[text_check]
        )
        button_c.click(fn=clear,outputs=[choices,text_check,button,button_c,file_output1])

        button_w1.click(
            fn=lambda file_input1: (
                "未上传文件，请上传所需的文件！" if not file_input1 else generate_file1(file_input1)
            ),
            inputs=[file_input1],
            outputs=[file_output1]
        )
    
    with gr.Tab(label="智能审查"):
        title = gr.Markdown("# 智能化审查服务")
        with gr.Column():  
            file_input2 = gr.File(label="上传文件")
            with gr.Row():
                button_base = gr.Button("生成项目基本信息")
                # button_clear = gr.Button("清空")
            base_info = gr.Textbox(label="项目基本信息", interactive=False)

            audit_re = gr.Textbox(label="审查要求")
            with gr.Row():
                btn = gr.Button("生成报告")
                btn_c = gr.Button("清空")

            text = gr.Textbox(label="检测报告",interactive=False)
            button_w2 = gr.Button("生成综合报告")
            file_output2 = gr.File(label="下载文件")

        button_base.click(
            fn=lambda file_input2: "未上传所需要生成信息的文件，请重新上传" if not file_input2 else generate_info(file_input2),
            inputs=[file_input2],
            outputs=[base_info]
        )
        # button_clear.click(fn=clear_all, outputs=[file_input2, base_info, audit_re, text, file_output2])
        btn.click(
            fn=lambda file_input2, audit_re, base_info: (
                "未上传所需要审查的报告，请重新上传" if not file_input2 else
                "项目基本信息为空，请先生成项目的基本信息！" if not base_info.strip() else
                get_audit(file_input2, audit_re, base_info)
            ),
            inputs=[file_input2, audit_re, base_info],
            outputs=[text]
        )
        btn_c.click(fn=clear, outputs=[audit_re,text,btn,btn_c,file_output2])
        button_w2.click(
            fn=lambda file_input2: "未上传所需要生成的文件，请重新上传" if not file_input2 else generate_file2(file_input2),
            inputs=[file_input2],
            outputs=[file_output2]
        )


    with gr.Row():
        with gr.Column(scale=2):
            gr.Button(visible=False) 
        with gr.Column(scale=3):
            gr.Image(value="https://raw.githubusercontent.com/wangqianxu7/-/main/logologo.png", elem_id="centered", width=500, show_label=False, show_download_button=False, container=False, show_fullscreen_button=False, mirror_webcam=False)
        with gr.Column(scale=1):
            gr.Button(visible=False) 

# 启用队列功能
demo.queue()
# 启动应用程序
demo.launch(share=True)
