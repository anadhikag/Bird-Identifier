"""Reusable Groq LLM client for the Bird Identification Assistant.

This module is responsible for ONE thing: all communication with the
Groq API. It is intentionally generic across two current use cases —
generating species.md knowledge base content, and (later) answering
RAG-retrieved questions in the chatbot — so that no other file in the
project needs to import the `groq` SDK directly.

Environment:
    GROQ_API_KEY must be set (e.g. via a .env file loaded with
    python-dotenv). A ValueError is raised at construction time if it is
    missing.
"""

from __future__ import annotations

import logging
import os
import re

from dotenv import load_dotenv
from groq import Groq, GroqError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "qwen/qwen3-32b"

# Matches <think>...</think> reasoning blocks that some reasoning models
# (including qwen/qwen3-32b) may still emit despite system-prompt
# instructions not to. Used as a defensive cleanup safeguard, not as a
# substitute for the prompt instructions.
_THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

# Matches a response fully wrapped in a markdown code fence, e.g.
# ```markdown\n...\n``` or ```\n...\n```, so it can be unwrapped if the
# model ignores the "no code fences" instruction.
_CODE_FENCE_WRAPPER_PATTERN = re.compile(
    r"^```(?:markdown)?\s*\n(.*)\n```\s*$", flags=re.DOTALL | re.IGNORECASE
)

SPECIES_MARKDOWN_TEMPLATE: str = """# {common_name}

## Basic Information
Scientific Name:
Family:
Order:

---

## Identification
Average Length:
Average Wingspan:
Average Weight:
Male vs Female Differences:
Distinctive Features:

---

## Habitat

---

## Geographic Distribution

---

## Migration
Migratory Status:
Migration Pattern:

---

## Diet

---

## Behaviour

---

## Vocalization

---

## Breeding

---

## Conservation
IUCN Status:
Threats:

---

## Ecological Importance

---

## Interesting Facts
- ...
- ...
- ...

---

## References
- Wikipedia – {common_name}
- Cornell Lab of Ornithology – {common_name}
- BirdLife International – {common_name}"""

_SPECIES_MARKDOWN_SYSTEM_PROMPT: str = """\
You are an ornithology reference writer producing factual, encyclopedia-\
quality content about bird species for a knowledge base.

Output discipline (follow these exactly, with no exceptions):
- Never reveal your reasoning.
- Never output <think> tags or anything resembling them.
- Never output analysis, planning, or meta-commentary about the task.
- Never explain your thought process before, during, or after the answer.
- Return ONLY the final markdown document, nothing else.
- Start your response immediately with the markdown heading ("# ...").
  The very first characters you output must be "# ".
- Do not wrap the response in markdown code fences (no ``` anywhere).
- Do not add any introductory text, sign-off, or notes after the
  markdown document ends.

Species name formatting:
- The heading and every other place the species name appears must use
  the correct, standard common name for the species, with correct
  ornithological capitalization, spacing, and hyphenation
  (for example: "Black-footed Albatross", not "Black footed Albatross"
  or "black-footed albatross").
- Ignore any numbering, underscores, or dataset-style formatting in the
  species name you are given (for example "001.Black_footed_Albatross")
  and output only the clean, correctly formatted common name.

Content rules:
- Be factual. Do not invent or guess specific facts, names, or numbers.
- Do not invent or estimate numerical measurements (length, wingspan,
  weight, incubation period, clutch size, lifespan, or any other
  measurement). If you are not confident of the exact or well-documented
  figure, write exactly "Unknown" instead of estimating or guessing.
- If any other fact is uncertain or you do not confidently know it,
  write exactly "Unknown" for that field instead of hallucinating.
- Write the scientific name as plain text with no italics, no asterisks,
  and no other markdown emphasis (for example: "Scientific Name:
  Phoebastria nigripes").
- In "Interesting Facts", replace each "..." placeholder with one
  concise, factual bullet point (three bullets total).
- In "References", replace each placeholder species name with the
  correctly formatted common name of the species, keeping the
  "Source - Species Name" format exactly as given.
- Use concise paragraphs, not filler language.
- Keep the entire document under 700 words.
- Follow the exact structure and heading order given in the template.
  Do not add, remove, reorder, or rename any section or field."""

_RAG_SYSTEM_PROMPT: str = """\
You are the answering component of a bird identification assistant.
The bird species has already been identified by a CNN image classifier,
so you must NOT question or attempt to re-identify the species.

Answer the user's question using ONLY the retrieved context provided to
you. Do not use outside knowledge and do not guess.

If the answer is not present in the retrieved context, say plainly that
you do not know, rather than speculating.

Respond in plain text only. Do not use markdown formatting."""


class GroqClient:
    """Thin, reusable wrapper around the Groq chat completions API.

    The underlying `groq.Groq` client is initialized exactly once per
    `GroqClient` instance and reused across all method calls, so
    instances of this class should be created once (e.g. as a module- or
    application-level singleton) and shared rather than recreated per
    request.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        """Initialize the Groq client.

        Loads GROQ_API_KEY from the environment (via a .env file if
        present) and constructs the underlying Groq SDK client once.

        Args:
            model: Groq model identifier to use for all completions
                issued by this instance.

        Raises:
            ValueError: If GROQ_API_KEY is not set in the environment.
        """
        load_dotenv()

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file or "
                "environment variables before creating a GroqClient."
            )

        self.model = model
        self._client = Groq(api_key=api_key)
        logger.info("GroqClient initialized with model: %s", self.model)

    @staticmethod
    def _clean_model_output(text: str) -> str:
        """Strip reasoning artifacts the model should not have produced.

        This is a defensive safeguard, not a replacement for the system
        prompt's "no reasoning, no code fences" instructions. Reasoning
        models such as qwen/qwen3-32b can occasionally leak `<think>`
        blocks or wrap output in a markdown code fence despite being
        told not to; this method removes both if present.

        Args:
            text: Raw text content returned by the model.

        Returns:
            Cleaned text with any `<think>...</think>` blocks removed
            and any wrapping code fence unwrapped, then stripped of
            leading/trailing whitespace.
        """
        cleaned = _THINK_TAG_PATTERN.sub("", text).strip()

        fence_match = _CODE_FENCE_WRAPPER_PATTERN.match(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        return cleaned

    def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> str:
        """Issue a single chat completion request to Groq.

        Centralizes request construction and error handling so both
        public methods share identical, well-tested call behavior.

        Args:
            system_prompt: System-role instructions for the model.
            user_prompt: User-role message content.
            temperature: Sampling temperature. Lower values produce more
                deterministic, factual output.

        Returns:
            The stripped text content of the model's response.

        Raises:
            RuntimeError: If the Groq API call fails or returns an
                unusable response.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
        except GroqError as error:
            logger.error("Groq API request failed: %s", error)
            raise RuntimeError(f"Groq API request failed: {error}") from error
        except Exception as error:  # noqa: BLE001 - surface any unexpected failure
            logger.error("Unexpected error calling Groq API: %s", error)
            raise RuntimeError(f"Unexpected error calling Groq API: {error}") from error

        if not response.choices:
            raise RuntimeError("Groq API returned no completion choices.")

        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Groq API returned an empty message content.")

        return self._clean_model_output(content)

    def generate_species_markdown(self, species_name: str) -> str:
        """Generate a full species.md knowledge base document.

        Args:
            species_name: Human-readable common name of the species,
                e.g. "Black footed Albatross".

        Returns:
            Markdown text following the fixed species.md template
            exactly, with no surrounding commentary or code fences.

        Raises:
            ValueError: If species_name is empty or blank.
            RuntimeError: If the Groq API call fails.
        """
        if not species_name or not species_name.strip():
            raise ValueError("species_name must be a non-empty string.")

        species_name = species_name.strip()
        logger.info("Generating species markdown for: %s", species_name)

        user_prompt = (
            f"Write the knowledge base document for the bird species given "
            f"below. The name may contain numbering, underscores, or "
            f"inconsistent capitalization from a dataset — correct it to "
            f"the standard common name before using it anywhere in the "
            f"document.\n\n"
            f"Species (raw input, may be poorly formatted): {species_name}\n\n"
            f"You must fill in the following markdown template exactly, "
            f"replacing only the blank fields, section bodies, and "
            f"placeholders with factual content. Keep every heading, field "
            f"label, and the '---' separators exactly as shown. Do not "
            f"wrap your answer in code fences.\n\n"
            f"Template:\n{SPECIES_MARKDOWN_TEMPLATE.format(common_name=species_name)}"
        )

        markdown = self._complete(
            system_prompt=_SPECIES_MARKDOWN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.3,
        )

        logger.info("Successfully generated markdown for: %s", species_name)
        return markdown

    def answer_question(
        self,
        species: str,
        retrieved_context: str,
        question: str,
    ) -> str:
        """Answer a user question using RAG-retrieved context.

        This method assumes the bird species has already been identified
        upstream by the CNN classifier; it does not attempt to identify
        or verify the species itself.

        Args:
            species: Common name of the already-identified bird species.
            retrieved_context: Text retrieved from the knowledge base
                that is relevant to the question.
            question: The user's natural-language question.

        Returns:
            Plain-text answer grounded strictly in retrieved_context. If
            the context does not contain the answer, the response states
            that the answer is not known.

        Raises:
            ValueError: If species, retrieved_context, or question is
                empty or blank.
            RuntimeError: If the Groq API call fails.
        """
        if not species or not species.strip():
            raise ValueError("species must be a non-empty string.")
        if not retrieved_context or not retrieved_context.strip():
            raise ValueError("retrieved_context must be a non-empty string.")
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string.")

        logger.info("Answering question for species '%s': %s", species, question)

        user_prompt = (
            f"Identified species: {species}\n\n"
            f"Retrieved context:\n{retrieved_context}\n\n"
            f"Question: {question}"
        )

        answer = self._complete(
            system_prompt=_RAG_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.2,
        )

        logger.info("Successfully generated answer for species '%s'.", species)
        return answer