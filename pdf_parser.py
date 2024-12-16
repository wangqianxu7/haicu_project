import pdfplumber
import pandas as pd
# from types import SimpleNamespace

# import pdfplumber
from langchain.docstore.document import Document
from tqdm import tqdm, trange
from loguru import logger
import fitz
import numpy as np
# from PIL import Image
# import multiprocessing
# 目前的问题:
# 1. 没有处理图片
# 2. 表格和文字会重复输入; 这个有点问题;(finish 远近匹配的问题)
# 3. 遇到更加复杂的情况, 比如图片等会遇到更严重的问题 
# 4. 目前解析有一点损失;



def rotate_pdf(input_pdf, output_pdf, rotation_angle=90):
    doc = fitz.open(input_pdf)
    for idx,page in enumerate(doc):
        print(f"正在处理第 {idx+1} 页")
        page.set_rotation(rotation_angle)
    doc.save(output_pdf)
    doc.close()


def check_text_orientation(pdf_path, output_pdf):
    doc = fitz.open(pdf_path)
    landcape_pages_idx = []
    portrait_pages_idx = []
    for page_number in range(len(doc)):
        # if page_number !=48:
        #     continue
        page = doc.load_page(page_number)
        text_blocks = page.get_text("blocks")
        if len(text_blocks) <= 4:
            y_score = 1
        else:
            y_list = [block[1] for block in text_blocks][1:-1]
            idx_list = [block[5] for block in text_blocks][1:-1]
            y_score =  np.corrcoef(y_list, idx_list)[0, 1]
        if y_score > 0.7:
            portrait_pages_idx.append(page_number)
        else:
            page.set_rotation(90)
            landcape_pages_idx.append(page_number)
    doc.save(output_pdf)
    doc.close()
    logger.info(f"横向页面：{landcape_pages_idx}")
    logger.info(f"纵向页面：{portrait_pages_idx}")
    return landcape_pages_idx, portrait_pages_idx

class PDFParser:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pdf = None
        self.page_content_dict = {}

    def __enter__(self):
        self.pdf = pdfplumber.open(self.pdf_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pdf.close()

    # def parse_page(self, page_number):
    #     page = self.pdf.pages[page_number]
    #     content = {
    #         'text': self.extract_text(page),
    #         'tables': self.extract_tables(page),
    #         'images': self.detect_images(page),
    #         'page_info': self.page_info(page)
    #     }
    #     return content

    def get_title(self):
        # 取路径的最后一个部分作为标题
        return self.pdf_path.split('/')[-1]


    def extract_text(self, page):
        return page.extract_text()
    
    def extract_tables(self, page):
        tables = page.extract_tables()
        parsed_tables = []
        for table in tables:
            table_string = ''
            first_row = table[0] #to location and replace
            last_row = table[-1] #to location and replace
            df = pd.DataFrame(table)
            df.fillna(method='ffill', axis=0, inplace=True)
            df.fillna(method='bfill', axis=0, inplace=True)
            df = df.applymap(lambda x: x.replace('\n', ' ') if isinstance(x, str) and '\n' in x else x)

            original_df = df.copy()
            for col in df.columns:
                for index in range(1, len(df)):
                    if original_df.at[index, col] == original_df.at[index-1, col] and len(original_df.at[index, col]) > 20:
                        df.at[index, col] = "同上"


            for index, row in df.iterrows():
                cleaned_row = row.fillna('None').tolist()  # 这里假设你仍然需要将None值转换为字符串'None'
                table_string += ('|' + '|'.join(cleaned_row) + '|' + '\n')
            table_string = table_string[:-1]


            parsed_tables.append((table_string, first_row, last_row))
        return parsed_tables
    
    def detect_images(self, page):
        images = page.images
        image_positions = [{'x0': img['x0'], 'y0': img['y0'], 'x1': img['x1'], 'y1': img['y1']} for img in images]
        return image_positions

    def page_info(self, page):
        return {'number': page.page_number, 'width': page.width, 'height': page.height}


    def parse_all_pages(self):
        txt = ""
        for page_number in trange(len(self.pdf.pages)):
            page = self.pdf.pages[page_number]
            page_content = self.extract_text(page)
            for (table, first_row, last_row) in self.extract_tables(page):
                first_row_str = next((item for item in first_row if item is not None and item != ""), "")
                first_row_str = first_row_str.split("\n")[0] if first_row_str else ""
                last_row_str = next((item for item in reversed(last_row) if item is not None and item != ""), "")
                last_row_str = last_row_str.split("\n")[-1] if last_row_str else ""
                first_row_pos = page_content.find(first_row_str)
                last_row_pos = page_content.rfind(last_row_str)

                if first_row_pos == -1 or last_row_pos == -1:
                    page_content += "####TABLE: {"+ table +"}\n"
                else:
                    page_content = page_content[:first_row_pos] +"####TABLE: {"+ table +"}\n" + page_content[last_row_pos+len(last_row):]
                page_content += "\n"
            self.page_content_dict[page_number] = page_content
            txt += page_content
        return Document(page_content=txt, metadata={"source":  self.pdf_path}), self.page_content_dict
