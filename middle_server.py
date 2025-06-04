import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from pydantic import BaseModel
import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
import os
import json
import requests
from enum import Enum

class BadReason(Enum):
    model_deleting_current_edit_row = "model_deleting_current_edit_row"
    lack_region_marker = "lack_region_marker"
    delete_too_much_row = "delete_too_much_row"

# INTERNAL_COMPLETION_API_URL = "http://7.216.58.118:8016/v1/completions"  # 替换为实际内部API地址

INTERNAL_COMPLETION_API_URL = (
    "http://localhost:8000/v1/completions"  # 替换为实际内部API地址
)
# model is stored at /mnt/vdc/models/modelscope/models/zeta_20250418_step1_lora_zeta_pretrain_merge/ on 7.216.58.118
# INTERNAL_COMPLETION_API_URL = "http://localhost:7878/v1/completions"  # 替换为实际内部API地址

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
default_log = os.path.join(current_directory, "single_log_model.jsonl")
default_delete_log = os.path.join(current_directory, "single_log_model_delete.jsonl")
# 获取当前文件所在的目录
current_directory = os.path.dirname(current_file_path)
START_MARKER = "<|editable_region_start|>"
END_MARKER = "<|editable_region_end|>"
CURSOR_MARKER = "<|user_cursor_is_here|>"

def re_sft_zeta_only_cleaned_pretrain_2k_zeta_spec_model_7b_openai(
    user_input, max_tokens=1500, temperature=0.0
):
    url = INTERNAL_COMPLETION_API_URL

    payload = json.dumps(
        {
            "model": "Qwen2.5-Coder-32B-Instruct",
            "prompt": user_input,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["<|editable_region_end|>"],
            "include_stop_str_in_output": True,
        }
    )
    headers = {"Content-Type": "application/json"}

    response = requests.request("POST", url, headers=headers, data=payload)
    rsp = response.json()
    return rsp

app = FastAPI()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义 OpenAI 风格的请求模型
class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "gpt-3.5-turbo"
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = 900
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[int, float]] = None
    user: Optional[str] = None

# 内部 Completion 接口的请求模型
class InternalCompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.0
    stop: Optional[List[str]] = None
    include_stop_str_in_output: bool = True

# 内部 Completion 接口的响应模型
class InternalCompletionResponse(BaseModel):
    text: str
    status: str

# 配置
TIMEOUT = 30.0  # 请求超时时间

def call_internal_completion(
    model: str, prompt: str, max_tokens: int = 900, stop: List[str] = None
) -> str:
    """
    同步调用内部 Completion 接口
    """
    try:
        rsp = re_sft_zeta_only_cleaned_pretrain_2k_zeta_spec_model_7b_openai(
            user_input=prompt, max_tokens=1500
        )
        return rsp
    except requests.HTTPError as e:
        logger.error(f"Internal API error: {e.response.text}")
        raise HTTPException(
            status_code=502, detail=f"Internal API error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Error calling internal API: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error calling internal API: {str(e)}"
        )

def extract_user_message(messages: List[ChatMessage]) -> str:
    """
    从消息列表中提取最后一条用户消息
    """
    user_messages = [msg.content for msg in messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(
            status_code=400, detail="No user message found in the conversation"
        )
    return user_messages[-1]  # 返回最后一条用户消息

def create_openai_response(
    completion_rsp: dict, request: ChatCompletionRequest
) -> Dict[str, Any]:
    """
    创建 OpenAI 风格的响应
    """
    completion_text = completion_rsp["choices"][0]["text"]

    return {
        "id": "chatcmpl-" + "".join([str(hash(completion_text))[:8]]),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion_text},
                "finish_reason": "stop",
            }
        ],
        "usage": completion_rsp.get("usage")
    }

def log_request_and_response(
    prompt,
    response: Dict[str, Any],
    start_time: float,
    end_time: float,
    log_file=default_log,
    additional_map: dict = None,
):
    """
    记录请求和响应到日志文件
    """
    log_entry = {
        "request_time": datetime.fromtimestamp(start_time).strftime("%Y-%m-%d:%H%M%S"),
        "response_time": datetime.fromtimestamp(end_time).strftime("%Y-%m-%d:%H%M%S"),
        "input": prompt,
        "output": response["choices"][0]["message"]["content"],
        "duration": end_time - start_time,
    }
    if additional_map:
        log_entry.update(additional_map)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

def count_lines_between_markers(
    input_string, original_start_marker=START_MARKER, original_end_marker=END_MARKER
):
    lines = input_string.split("\n")
    start_line_number = None
    end_line_number = None

    for i, line in enumerate(lines):
        if original_start_marker in line:
            start_line_number = i
        if original_end_marker in line:
            end_line_number = i
            break  # 只需要找到第一个匹配的结束标记

    if start_line_number is None or end_line_number is None:
        raise ValueError("One or both markers not found in the input string.")

    return end_line_number - start_line_number

def is_model_deleting_current_edit_row(input_str: str, output_str: str) -> bool:
    def extract_content(s: str, start_marker: str, end_marker: str) -> str:
        start_index = s.find(start_marker) + len(start_marker)
        end_index = s.find(end_marker)
        return s[start_index:end_index].strip()

    def remove_cursor_line(content: str, cursor_marker: str) -> str:
        lines = content.split("\n")
        new_lines = [line for line in lines if cursor_marker not in line]
        return "\n".join(new_lines)

    # Extract content from input and output
    x = extract_content(input_str, START_MARKER, END_MARKER)
    y = extract_content(output_str, START_MARKER, END_MARKER)

    # Remove the line containing the cursor marker from x
    new_x = remove_cursor_line(x, CURSOR_MARKER)

    is_deleting_current_edit_row = new_x == y
    # Compare new_x with y
    return is_deleting_current_edit_row

@app.post("/v1/chat/completions")
async def chat_completion(request: Request, body: ChatCompletionRequest):
    """
    处理 OpenAI 风格的 ChatCompletion 请求
    """
    client_host = request.client.host
    user_agent = request.headers.get("User-Agent", "")
    additional_map = {}
    additional_map["client_host"] = client_host
    additional_map["user_agent"] = user_agent
    print(f"request: {request}")
    print(f"body: {body}")
    stream = body.stream
    print(f"stream: {stream}")

    if stream:
        user_message = extract_user_message(body.messages)
        payload = {
            "model":"Qwen2.5-Coder-32B-Instruct", 
            "prompt": user_message,
            "max_tokens": 1500,
            "temperature": 0,
            "stop": ["<|editable_region_end|>"],
            "include_stop_str_in_output": True,
            "stream": True
        }
        response = requests.post(INTERNAL_COMPLETION_API_URL, json=payload, stream=True)
        return StreamingResponse(response, media_type="text/event-stream")

    BAD_REASON_KEY = "bad_reason"
    try:
        # 提取用户消息
        start_time = time.time()
        user_message = extract_user_message(body.messages)
        logger.info(f"Processing user message: {user_message[:100]}...")

        # 调用内部 Completion 接口
        completion_rsp = call_internal_completion(
            model="Qwen2.5-Coder-32B-Instruct",
            prompt=user_message,
            max_tokens=body.max_tokens,
            stop=body.stop,
        )

        # 构建 OpenAI 风格的响应
        end_time = time.time()
        response = create_openai_response(completion_rsp, body)

        output = response["choices"][0]["message"]["content"]
        try:
            input_region_row_num = count_lines_between_markers(user_message)
            output_region_row_num = count_lines_between_markers(output)
            if input_region_row_num - output_region_row_num >= 3:
                additional_map[BAD_REASON_KEY] = BadReason.delete_too_much_row.value
                log_request_and_response(
                    user_message,
                    response,
                    start_time,
                    end_time,
                    default_delete_log,
                    additional_map=additional_map,
                )
                # ! 屏蔽删除太多行
                response["choices"][0]["message"]["content"] = user_message
            else:
                if is_model_deleting_current_edit_row(user_message, output):
                    response["choices"][0]["message"]["content"] = (
                        user_message  # 如果模型推荐删除用户正在写的行，就返回源输入内容
                    )
                    additional_map[BAD_REASON_KEY] = (
                        BadReason.model_deleting_current_edit_row.value
                    )
                log_request_and_response(
                    user_message,
                    response,
                    start_time,
                    end_time,
                    additional_map=additional_map,
                )

        except:
            bad_region_log = os.path.join(
                current_directory, "single_log_model_delete_bad_marker.jsonl"
            )
            additional_map[BAD_REASON_KEY] = BadReason.lack_region_marker.value
            log_request_and_response(
                user_message,
                response,
                start_time,
                end_time,
                bad_region_log,
                additional_map=additional_map,
            )

        return JSONResponse(content=response)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8100)
