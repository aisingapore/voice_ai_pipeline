import asyncio
from enum import Enum
import os
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def main(room_url: str, token: str):
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.frames.frames import EndFrame, LLMMessagesFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.services.xtts import XTTSService
    from pipecat.services.openai import OpenAILLMService
    from pipecat.transports.services.daily import DailyParams, DailyTransport
    from pipecat.transcriptions.language import Language
    from loguru import logger
    from pipecat.frames.frames import (
        ErrorFrame,
        Frame,
        TranscriptionFrame,
    )
    from pipecat.services.ai_services import SegmentedSTTService
    from pipecat.utils.time import time_now_iso8601
    from typing import AsyncGenerator
    import io
    import wave
    import numpy as np
    class Model(Enum):
        """Available OpenAI Whisper API models"""

        WHISPER_1 = "whisper-1"

    class WhisperAPIService(
        SegmentedSTTService
    ):  # override SegmentedSTTService from Pipecat
        # Unusual to be placed here, expected for these changes to be in the Whisper server, though I found it simpler to do this for now to fit the Pipecat interface.
        """Service for OpenAI's Whisper API transcription"""

        def __init__(
            self,
            *,
            api_key: str,
            model: str | Model = Model.WHISPER_1,
            base_url: str = "https://api.openai.com/v1",
            language: Language = None,
            temperature: float = 0,
            prompt: str = None,
            response_format: str = "json",
            sample_rate: int = 16000,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self._api_key = api_key
            self._base_url = base_url.rstrip("/")
            self._session: aiohttp.ClientSession | None = None
            self._sample_rate = sample_rate
            self.set_model_name(model if isinstance(model, str) else model.value)

            self._settings = {
                "model": self.model_name,
                "temperature": temperature,
                "response_format": response_format,
            }

            if language:
                self._settings["language"] = self.language_to_service_language(language)
            if prompt:
                self._settings["prompt"] = prompt

        def can_generate_metrics(self) -> bool:
            return True

        # TODO: this is model specific, so check if it is compatible with the model you are using
        def _prepare_audio(self, audio: bytes) -> bytes:
            """Convert raw PCM audio to WAV format"""
            # Convert to float32 array first (matching whisper.py)
            audio_float = (
                np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            )

            # Convert back to int16 for WAV
            audio_int16 = (audio_float * 32768).astype(np.int16)

            # Create WAV
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, "wb") as wav_file:
                    wav_file.setnchannels(1)  # mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(self._sample_rate)
                    wav_file.writeframes(audio_int16.tobytes())
                return wav_io.getvalue()

        async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
            """Transcribes given audio using Whisper API"""
            try:
                await self.start_processing_metrics()
                await self.start_ttfb_metrics()

                # Prepare audio in the same way as local Whisper
                wav_audio = self._prepare_audio(audio)

                # Create session for this request
                async with aiohttp.ClientSession(
                    headers={"Authorization": f"Bearer {self._api_key}"}
                ) as session:
                    # Prepare form data
                    form = aiohttp.FormData()
                    form.add_field(
                        "file",
                        wav_audio,
                        filename="audio.wav",
                        content_type="audio/wav",
                    )

                    # Add settings as form fields
                    for key, value in self._settings.items():
                        if value is not None:
                            form.add_field(key, str(value))

                    async with session.post(
                        f"{self._base_url}/audio/transcriptions", data=form
                    ) as response:
                        await self.stop_ttfb_metrics()

                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(
                                f"Whisper API error: {response.status} - {error_text}"
                            )
                            yield ErrorFrame(f"Transcription failed: {error_text}")
                            return

                        result = await response.json()
                        text = result.get("text", "").strip()

                        if text:
                            logger.debug(f"Transcription: [{text}]")
                            yield TranscriptionFrame(text, "", time_now_iso8601())

            except Exception as e:
                logger.exception(f"Whisper API error: {e}")
                yield ErrorFrame(f"Transcription failed: {str(e)}")
            finally:
                await self.stop_processing_metrics()

        def language_to_service_language(self, language: Language) -> str:
            """Convert internal language enum to ISO-639-1 codes"""
            return str(language.value).split("-")[0].lower()

    class SealionLLMService(OpenAILLMService):
        """A service for interacting with any OpenAI-compatible interface.

        This service extends OpenAILLMService to connect to SEA-LION's API endpoint while
        maintaining full compatibility with OpenAI's interface and functionality.

        Args:
            api_key (str): The API key for accessing the LLM API
            base_url (str, optional): The base URL for the LLM API.
            model (str, optional): The model identifier to use.
            **kwargs: Additional keyword arguments passed to OpenAILLMService
        """

        def __init__(
            self,
            *,
            api_key: str,
            base_url: str = "https://api.sea-lion.ai/v1",
            model: str = "aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct",
            **kwargs,
        ):
            super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)

        def create_client(self, api_key=None, base_url=None, **kwargs):
            """Create OpenAI-compatible client for SEALION API endpoint."""
            logger.debug(f"Creating SEALION client with api {base_url}")
            return super().create_client(api_key, base_url, **kwargs)

    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            token,
            "bot",
            DailyParams(
                audio_out_enabled=True,
                audio_out_sample_rate=24000,  # TODO: be careful with this setting, it is model specific e.g. audio_out_sample_rate should match the sample rate specified by the specifications of the STT model
                transcription_enabled=False,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                vad_audio_passthrough=True,
            ),
        )

        # stt = WhisperAPIService(api_key=os.getenv("OPENAI_API_KEY"), model="whisper-1") # for using OpenAI's Whisper API
        stt = WhisperAPIService(
            api_key=os.getenv(
                "WHISPER_API_KEY"
            ),  # replace with your own API key for the Whisper server
            base_url="http://35.91.186.23:8000/v1",  # TODO: replace with your own base URL for the Whisper server
            model="whisper-1",
        )

        # tts = CartesiaTTSService(
        #     api_key=os.getenv("CARTESIA_API_KEY", ""), voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22"
        # ) # for using Cartesia's API
        tts = XTTSService(
            aiohttp_session=session,
            voice_id="Ana Florence",  # Marcos Rudaski
            language=Language.EN,
            base_url="http://35.94.29.191:8000",  # TODO: replace with your own base URL for the XTTS server
        )

        # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o") # for using OpenAI's API
        llm = SealionLLMService(
            api_key=os.getenv("AISG_API_KEY"),
        )  # for using SEA-LION's API

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        # Event handlers for Daily WebRTC
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await transport.capture_participant_transcription(participant["id"])
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([LLMMessagesFrame(messages)])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.queue_frame(EndFrame())

        runner = PipelineRunner()

        await runner.run(task)


def _voice_bot_process(room_url: str, token: str):
    asyncio.run(main(room_url, token))
