
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from loguru import logger
import requests
import json


def clear_lines():
    print('\033[2J')


class ChatQwen:
    def __init__(self, model_name="/share/home/tj24147/data/huggingface_model/Qwen/Qwen2-72B-Instruct-GPTQ-Int4", device="cuda"):
        """
        Initialize the ChatQwen class with a model and tokenizer.
        
        Parameters:
        - model_name: The name of the model to load.
        - device: The device to load the model onto.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     torch_dtype="auto",
        #     device_map="auto",
        #     trust_remote_code=True
        # )
        # print("self.model.config.use_flash_attn", self.model.config.use_flash_attn)
        # print("model.config.use_logn_attn", self.model.config.use_logn_attn)
        

        # self.model.eval()  # Set the model to evaluation mode
        self.max_length = 8192
        self.top_p = 0.9
        self.temperature = 0.2

    def chat(self, query, history=[]):
        """
        Generate a chat response based on the given query and history.
        
        Parameters:
        - query: The user input query.
        - history: The chat history.
        
        Returns:
        - response: The generated text response.
        - history: The updated chat history including the current query.
        """

        response=requests.post('http://localhost:8080/chat',json={
            'query':query,
            'stream': True,
            'history':history,
        },stream=True)

        text = ''
        for chunk in response.iter_lines(chunk_size=8192,decode_unicode=False,delimiter=b"\0"):
            if chunk:
                data=json.loads(chunk.decode('utf-8'))
                text =data["text"].rstrip('\r\n') # 确保末尾无换行
                # 打印最新内容
                # print(text)

        # logger.debug(f"response: {text}")
        # history.append((query,text))
        return text, history
    
    
    def stream_chat(self, query, history=[]):
        response=requests.post('http://localhost:8080/chat',json={
            'query':query,
            'stream': True,
            'history':history,
        },stream=True)
        text = ''
        for chunk in response.iter_lines(chunk_size=8192,decode_unicode=False,delimiter=b"\0"):
            if chunk:
                data=json.loads(chunk.decode('utf-8'))
                text=data["text"].rstrip('\r\n') # 确保末尾无换行
                # 打印最新内容
                yield text, history
        # logger.debug(f"response: {text}")
        # history.append((query,text))
        yield text, history
        
       
 
class ChatGLM:
    def __init__(self, model_name="/home/admin/data/huggingface_model/chatglm3-6b", device="cuda"):
        """
        Initialize the ChatQwen class with a model and tokenizer.
        
        Parameters:
        - model_name: The name of the model to load.
        - device: The device to load the model onto.
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()  # Set the model to evaluation mode
        self.max_length = 4096
        self.top_p = 0.9
        self.temperature = 0.7

    def chat(self, query, history=[]):
       '''
       def chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
             max_length: int = 8192, num_beams=1, do_sample=True, top_p=0.8, temperature=0.8, logits_processor=None,**kwargs):
       '''
       response, history = self.model.chat(self.tokenizer, query, history=history)
       return response, history
    

    def stream_chat(self, query, history=[]):
        '''
        def stream_chat(self, tokenizer, query: str, history: List[Dict] = None, role: str = "user",
                    past_key_values=None,max_length: int = 8192, do_sample=True, top_p=0.8, temperature=0.8,
                    logits_processor=None, return_past_key_values=False, **kwargs):
        return 
        response (str): 模型的响应文本。
        history (List[Dict]): 更新后的对话历史。
        past_key_values (List[Tensor], 可选): 如果return_past_key_values为True，返回用于下一步生成的过去的键值对。
        role: 'user' or 'assistant'
        
        '''
        yield from self.model.stream_chat(tokenizer=self.tokenizer,
                                            query=query,
                                            history=history,
                                            max_length=self.max_length,
                                            top_p=self.top_p,
                                            temperature=self.temperature)
    
        
if __name__ == "__main__":
    model = ChatQwen()
    query = "你叫什么名字？"
    response, history = model.chat(query)
    print(response)