import os
from typing import Any, Optional, Union

from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import (
    ChatCompletionPrompt,
    CompletionPrompt,
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
    CrosshatchChatMessage,
    CrosshatchCreateChatPrompt,
    CrosshatchChatCompletionPrompt
)
from evals.record import record_sampling

import requests
import json
import time
import random
from tenacity import retry, stop_after_attempt, wait_random

import openai


class LangChainLLMCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class RouterCompletionFn(CompletionFn):
    def __init__(self, llm_kwargs: Optional[dict] = None, **kwargs) -> None:
        self.router_url = 'http://localhost:3000/api/decent/chat/completions'
        self.headers = {
            'Content-Type': 'application/json'
        }
        self.router = llm_kwargs['router']
        print(self.router)

    def call_router(self, prompt: str) -> str:
        # 
        data = {
            "messages": prompt,
            "CompletionProps": {
                "router": self.router
            }
        }
        print(data)
        response = self._post_request(data)
        answer = response.content.decode("utf-8")
        if response.status_code != 200:
            raise Exception('Router error.')
        return answer

    @retry(stop=stop_after_attempt(3))
    def _post_request(self, data):
        response = requests.post(self.router_url, headers=self.headers, json=data)
        return response


    def __call__(self,
                 prompt: Union[str, CrosshatchCreateChatPrompt],
                 **kwargs) -> LangChainLLMCompletionResult:
        if not isinstance(prompt, Prompt):
            assert (
                isinstance(prompt, str)
                or (isinstance(prompt, list) and all(isinstance(token, int) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(token, str) for token in prompt))
                or (isinstance(prompt, list) and all(isinstance(msg, dict) for msg in prompt))
            ), f"Got type {type(prompt)}, with val {type(prompt[0])} for prompt, expected str or list[int] or list[str] or list[dict[str, str]]"

            prompt = CrosshatchChatCompletionPrompt(
                raw_prompt=prompt,
            )
        crosshatch_create_prompt: CrosshatchCreateChatPrompt = prompt.to_formatted_prompt()
        print(crosshatch_create_prompt)
        # prompt = CompletionPrompt(prompt).to_formatted_prompt()
        response = self.call_router(crosshatch_create_prompt)
        record_sampling(prompt=prompt, sampled=response)
        return LangChainLLMCompletionResult(response)

