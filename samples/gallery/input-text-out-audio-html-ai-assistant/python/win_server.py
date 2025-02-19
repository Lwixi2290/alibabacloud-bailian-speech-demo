#!/usr/bin/env python3
# Copyright (c) alibaba.. All Rights Reserved.
# MIT License  (https://opensource.org/licenses/MIT)
import asyncio
import json
import os
import queue
import sys
from http import HTTPStatus

import dashscope
import websockets
from dashscope import Generation
from dashscope.audio.tts_v2 import *

# Windows系统需要特殊设置
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# API密钥配置
if 'DASHSCOPE_API_KEY' in os.environ:
    dashscope.api_key = os.environ['DASHSCOPE_API_KEY']
else:
    dashscope.api_key = 'api-key'  # 替换为你的有效API密钥

messages = []  # 对话历史记录

class WSCallback(ResultCallback):
    def __init__(self, websocket, loop: asyncio.AbstractEventLoop):
        self.websocket = websocket
        self.loop = loop
        self.completed = queue.Queue()

    def on_open(self):
        print('WebSocket连接已建立')

    def on_complete(self):
        print('语音合成任务完成')
        self.completed.put(True)

    def on_error(self, message: str):
        print(f'语音合成错误: {message}')
        self.completed.put(False)

    def on_close(self):
        print('WebSocket连接已关闭')

    def on_data(self, data: bytes) -> None:
        self.loop.call_soon_threadsafe(
            asyncio.create_task,
            self.websocket.send(data)
        )

async def LlmTask(query, websocket):
    global messages
    loop = asyncio.get_event_loop()
    
    # 初始化语音合成器
    synthesizer_callback = WSCallback(websocket=websocket, loop=loop)
    synthesizer = SpeechSynthesizer(
        model='cosyvoice-v1',
        voice='longxiaochun',
        format=AudioFormat.PCM_22050HZ_MONO_16BIT,
        callback=synthesizer_callback,
    )

    # 处理对话历史
    messages.append({'role': 'user', 'content': query})
    messages = messages[-10:]  # 保留最近10轮对话

    # 调用大模型生成回复
    assistant_response = ''
    try:
        responses = Generation.call(
            model='qwen-plus',
            messages=messages,
            result_format='message',
            stream=True,
            incremental_output=True,
        )

        for response in responses:
            if response.status_code == HTTPStatus.OK:
                text_chunk = response.output.choices[0]['message']['content']
                assistant_response += text_chunk
                print('生成内容:', text_chunk)
                
                # 发送文本片段
                await websocket.send(text_chunk)
                # 流式语音合成
                synthesizer.streaming_call(text_chunk)
                
            await asyncio.sleep(0.05)
    except Exception as e:
        print(f'生成错误: {str(e)}')
        await websocket.close()
    finally:
        synthesizer.async_streaming_complete()
        # 等待语音合成完成
        while synthesizer_callback.completed.empty():
            await asyncio.sleep(0.1)
        
        messages.append({'role': 'assistant', 'content': assistant_response})

async def handle_client(websocket, path):
    try:
        async for message in websocket:
            print('收到消息:', message)
            data = json.loads(message)
            await LlmTask(data['text'], websocket)
    except websockets.exceptions.ConnectionClosed:
        print("客户端断开连接")
    except Exception as e:
        print(f"处理错误: {str(e)}")

async def main():
    port = 11111
    server = await websockets.serve(
        handle_client,
        'localhost',
        port,
        ping_interval=None
    )
    print(f'服务器已启动: ws://localhost:{port}')
    await server.wait_closed()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("服务器已手动停止")
    except Exception as e:
        print(f"服务器异常: {str(e)}")
