from typing import AsyncGenerator
from app.core.logger import logger
from groq import AsyncGroq
from app.core.config import settings

LLM_MODEL_NAME = settings.LLM_MODEL_NAME
GROQ_API_KEY = settings.GROQ_API_KEY


class GroqService:
    def __init__(self, groq_client: AsyncGroq = None):
        self.groq_client = groq_client

    def initialize_client(self):
        """Initialize the Groq client with the API key from settings."""
        self.groq_client = AsyncGroq(api_key=GROQ_API_KEY)

    def get_client(self) -> AsyncGroq:
        """Get the initialized Groq client, initializing it if necessary."""
        if self.groq_client is None:
            self.initialize_client()
        return self.groq_client

    async def close_client(self):
        """Close the Groq client if it has been initialized."""
        if self.groq_client:
            await self.groq_client.close()
        logger.info("Groq client closed successfully.")

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
