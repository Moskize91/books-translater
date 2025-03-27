from typing import cast
from time import sleep
from pydantic import SecretStr
from langchain_core.language_models import LanguageModelInput
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from .error import is_retry_error


class LLMExecutor:
  def __init__(
    self,
    llm_model: ChatAnthropic | ChatOpenAI | ChatVertexAI,
    timeout: float,
    retry_times: int,
    retry_interval_seconds: float,
  ) -> None:
    self._llm_model: ChatAnthropic | ChatOpenAI | ChatVertexAI = llm_model
    self._timeout: float = timeout
    self._retry_times: int = retry_times
    self._retry_interval_seconds: float = retry_interval_seconds

  def request(self, input: LanguageModelInput) -> str:
    last_error: Exception | None = None
    result: str | None = None

    try:
      for i in range(self._retry_times + 1):
        try:
          response = self._llm_model.invoke(input)
        except Exception as err:
          last_error = err
          if not is_retry_error(err):
            raise err
          print(f"request failed with connection error, retrying... ({i + 1} times)")
          if self._retry_interval_seconds > 0.0 and \
            i < self._retry_times:
            sleep(self._retry_interval_seconds)
          continue

        try:
          result = str(response.content)
          break

        except Exception as err:
          last_error = err
          print(f"request failed with parsing error, retrying... ({i + 1} times)")
          if self._retry_interval_seconds > 0.0 and \
            i < self._retry_times:
            sleep(self._retry_interval_seconds)
          continue

    except KeyboardInterrupt as err:
      if last_error is not None:
        print(last_error)
      raise err

    if last_error is not None:
      raise last_error

    return cast(str, result)
