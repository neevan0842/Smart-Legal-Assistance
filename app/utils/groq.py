from typing import AsyncGenerator

from groq import AsyncGroq
from app.core.config import settings

LLM_MODEL_NAME = settings.LLM_MODEL_NAME


class GroqService:
    def __init__(self, groq_client: AsyncGroq):
        self.groq_client = groq_client

    async def generate_answer(self, query: str, system_prompt: str) -> str:
        """Generate an answer for a legal query using the provided context and system prompt."""

        chat_completion = await self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            model=LLM_MODEL_NAME,
        )

        return chat_completion.choices[0].message.content

    async def generate_answer_stream(
        self, query: str, system_prompt: str
    ) -> AsyncGenerator[str, None]:
        """Generate an answer for a legal query using the provided context and system prompt with streaming."""

        # Get streaming response from LLM
        stream = await self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            model=settings.LLM_MODEL_NAME,
            stream=True,
        )

        # Yield chunks as they arrive in SSE format
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                # Format as Server-Sent Events
                yield f"data: {content}\n\n"
        yield "data: [DONE]\n\n"
