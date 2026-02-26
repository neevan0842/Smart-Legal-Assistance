import json
from typing import AsyncGenerator, List
from pydantic import BaseModel
from app.core.logger import logger
from groq import AsyncGroq
from app.core.config import settings
from app.utils.utils import parse_relevance_score

LLM_MODEL_NAME = settings.LLM_MODEL_NAME
GROQ_API_KEY = settings.GROQ_API_KEY
EVALUATION_MODEL_NAME = settings.EVALUATION_MODEL_NAME


class RelavenceScore(BaseModel):
    score: List[int]


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

    async def generate_answer(self, prompt_messages: List[dict]) -> str:
        """Generate an answer for a legal query using the provided context and system prompt."""

        chat_completion = await self.groq_client.chat.completions.create(
            messages=prompt_messages,
            model=LLM_MODEL_NAME,
        )

        return chat_completion.choices[0].message.content

    async def generate_answer_stream(
        self, prompt_messages: List[dict]
    ) -> AsyncGenerator[str, None]:
        """Generate an answer for a legal query using the provided context and system prompt with streaming."""

        # Get streaming response from LLM
        stream = await self.groq_client.chat.completions.create(
            messages=prompt_messages,
            model=LLM_MODEL_NAME,
            stream=True,
        )

        # Yield chunks as they arrive in SSE format
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                # Format as Server-Sent Events
                yield f"data: {content}\n\n"
        yield "data: [DONE]\n\n"

    async def evaluate_rag_results_to_get_relevance_scores(
        self, query: str, retrieved_documents: List[str]
    ) -> List[int]:

        MAX_DOCS = 8
        if len(retrieved_documents) > MAX_DOCS:
            logger.warning(
                f"Truncating retrieved_documents from {len(retrieved_documents)} to {MAX_DOCS} to fit token limit."
            )
            retrieved_documents = retrieved_documents[:MAX_DOCS]
        documents_list = json.dumps(retrieved_documents)

        prompt = f"""
You are an expert relevance evaluator.

You will receive:
1) A user query
2) A list of retrieved documents (strings)

Your task:
For each document, in the order provided, assign a numeric relevance score from 0 to 3:
  • 0 = Not relevant
  • 1 = Slightly relevant
  • 2 = Relevant
  • 3 = Highly relevant

You MUST return ONLY valid JSON with this structure:

{{
  "score": [
    <number>,
    <number>,
    <number>,
    ...
  ]
}}

Make sure:
1) Each number is separated by a comma.
2) There are no quotes around numbers.
3) There are no extra keys or fields.
4) The count of numbers equals the count of documents.

Example:

Input:
Query:
"best way to learn Python"

Retrieved documents:
[
  "Article about Python basics",
  "Completely unrelated gardening tips",
  "Guide to advanced Python tricks"
]

Correct output:
{{
  "score": [3, 0, 2]   <-- note explicit commas and square brackets
}}

--------------------------
Now answer for the following:

Query:
"{query}"

Retrieved documents:
{documents_list}
"""

        response = await self.groq_client.chat.completions.create(
            model=EVALUATION_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and precise assistant for evaluating the relevance of retrieved documents to a user query.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "RelevanceScore",
                    "schema": RelavenceScore.model_json_schema(),
                },
            },
        )

        raw_result = json.loads(response.choices[0].message.content or "{}")
        result = RelavenceScore.model_validate(raw_result)
        parsed_scores = parse_relevance_score(result.score)
        logger.debug(f"Raw response from Groq relevance evaluation: {raw_result}")
        return parsed_scores
