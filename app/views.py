from django.http import HttpResponse
import time
import json
from transformers import AutoTokenizer, AutoModel
from accelerate import utils


# 加载ChatGLM模型，指定ChatGLM模型的绝对路径
chatglm_path = '/root/model/chatglm-6b'
tokenizer = AutoTokenizer.from_pretrained(chatglm_path, trust_remote_code=True, revision="v0.1.0")
# 以下是多GUP卡引入模型的方式，num_gpus表示GPU卡的个数
model = utils.load_model_on_gpus(chatglm_path, num_gpus=2)
# 以下是MacOS引入模型的方式
# model = AutoModel.from_pretrained(chatglm_path, trust_remote_code=True, revision="v0.1.0").float().to('mps')
model = model.eval()


def index(request):
    if request.method == 'GET':
        return HttpResponse('Successful !')
    if request.method == 'POST':
        json_param = json.loads(request.body.decode())
        question = json_param.get('question', 0)
        print(
            "开始询问ChatGLM:",
            time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(time.time())
            )
        )
        # response, history = model.chat(tokenizer, question, history=[]); print(response)
        response = question
        print(
            "结束询问ChatGLM:",
            time.strftime(
                "%Y-%m-%d %H:%M:%S",
                time.localtime(time.time())
            )
        )
        return HttpResponse('{"response": "' + response + '"}')
    return HttpResponse('{"response": "Timeout"}')
