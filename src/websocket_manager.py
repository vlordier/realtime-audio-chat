"""Manages WebSocket connection and communication with the OpenAI API."""

import asyncio
import base64
import json
import logging
import queue  # For thread-safe queues
import shutil

import websockets
import pyaudio

from src.audio_manager import AudioManager
from src.config_manager import ConfigManager

logger = logging.getLogger(__name__)

RETRY_LIMIT = 5
RETRY_DELAY = 5  # seconds


class WebSocketManager:
    """Handles WebSocket connection and communication."""

    def __init__(
        self, config_manager: ConfigManager, audio_manager: AudioManager
    ) -> None:
        """Initializes the WebSocketManager with configuration and audio managers.

        Args:
            config_manager (ConfigManager): The configuration manager instance.
            audio_manager (AudioManager): The audio manager instance.
        """
        self.config = config_manager
        self.audio_manager = audio_manager
        self.ws = None
        self.should_reconnect = True
        # Use a thread-safe queue to communicate between audio callback and async tasks
        self.audio_queue: queue.Queue = queue.Queue()
        # Start the input stream with the audio callback
        if not self.audio_manager.is_input_stream_active():
            self.audio_manager.start_input_stream(self.audio_callback)

        # Start the output stream
        if not self.audio_manager.is_output_stream_active():
            self.audio_manager.start_output_stream()

    async def connect(self) -> None:
        """Connects to the WebSocket server with retries and manages communication."""
        retries = 0
        while retries <= RETRY_LIMIT and self.should_reconnect:
            try:
                # Retrieve configuration parameters with validation
                base_url = self.config.get("base_url")
                model_type = self.config.get("model_type")
                api_key = self.config.get("api_key")

                if not base_url or not model_type or not api_key:
                    logger.error("Invalid configuration parameters.")
                    return

                uri = f"{base_url}?model={model_type}"

                extra_headers = {
                    "Authorization": f"Bearer {api_key}",
                    "OpenAI-Beta": "realtime=v1",
                }

                async with websockets.connect(uri, extra_headers=extra_headers) as ws:
                    self.ws = ws
                    logger.info("Connected to WebSocket successfully.")
                    await self.handle_communication()
                    break  # Exit the retry loop if connection is successful
            except (
                websockets.ConnectionClosedError,
                websockets.InvalidStatusCode,
            ) as e:
                logger.error("WebSocket error: %s", e)
                if retries < RETRY_LIMIT:
                    retries += 1
                    logger.info(
                        "Retrying connection in %d seconds... (%d/%d)",
                        RETRY_DELAY,
                        retries,
                        RETRY_LIMIT,
                    )
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    logger.error(
                        "Maximum retry limit reached. Could not connect to WebSocket."
                    )
                    self.should_reconnect = False
                    break
            except Exception as e:
                logger.exception("Unexpected error during connection: %s", e)
                self.should_reconnect = False
                break

    def audio_callback(self, in_data, frame_count, time_info, status_flags):
        """Callback function for processing audio input using pyaudio.

        Args:
            in_data (bytes): The input audio data.
            frame_count (int): The number of frames.
            time_info (dict): A dictionary containing timing information.
            status_flags (int): The status flags.
        Returns:
            tuple: (None, pyaudio.paContinue) to continue streaming.
        """
        try:
            # Put the audio data into the queue for the async task to consume
            self.audio_queue.put_nowait(in_data)
        except Exception as e:
            logger.exception("Failed to enqueue audio data: %s", e)
        return (None, pyaudio.paContinue)

    async def handle_communication(self) -> None:
        """Handles sending and receiving messages over WebSocket."""
        try:
            # Create tasks for receiving messages, sending audio, and displaying meters
            receive_task = asyncio.create_task(self.receive_messages())
            send_task = asyncio.create_task(self.send_audio_stream())
            meter_task = asyncio.create_task(self.display_audio_meters())

            # Wait for tasks to complete, handling exceptions
            await asyncio.gather(receive_task, send_task, meter_task)
        except Exception as e:
            logger.exception("Error in handle_communication: %s", e)
        finally:
            # Ensure all tasks are cancelled and resources are cleaned up
            receive_task.cancel()
            send_task.cancel()
            meter_task.cancel()
            await self.close()

    async def receive_messages(self) -> None:
        """Receives messages from the WebSocket."""
        try:
            async for message in self.ws:
                # Get terminal size for logging
                # columns, _ = shutil.get_terminal_size(fallback=(80, 24))
                # maxl = columns - 5
                # logger.info(
                #     message if len(message) <= maxl else (message[:maxl] + " ...")
                # )
                evt = json.loads(message)
                if evt["type"] == "session.created":
                    # logger.info("Connected: say something to GPT-4")
                    if not self.audio_manager.is_input_stream_active():
                        self.audio_manager.start_input_stream(self.audio_callback)
                elif evt["type"] == "response.audio.delta":
                    audio = base64.b64decode(evt["delta"])
                    self.audio_manager.write_output(audio)
                elif evt["type"] == "error":
                    logger.error("Error received: %s", evt["error"])
        except websockets.ConnectionClosedError as e:
            logger.error("Connection closed unexpectedly: %s", e)
        except Exception as e:
            logger.exception("Unexpected error in receive_messages: %s", e)
        # No need to call await self.close() here; it's handled in handle_communication

    async def send_audio_stream(self) -> None:
        """Continuously reads audio input and sends it over WebSocket."""
        try:
            # Start the input stream with the audio callback
            if not self.audio_manager.is_input_stream_active():
                self.audio_manager.start_input_stream(self.audio_callback)

            while True:
                # Get audio data from the queue in an asynchronous way
                in_data = await asyncio.to_thread(self.audio_queue.get)
                if in_data is None:
                    # Sentinel value to stop the loop
                    break

                # Compute input level for meters
                input_rms = self.audio_manager.compute_rms_level(in_data)
                self.audio_manager.set_input_level(input_rms)

                # Convert audio data to base64
                audio_base64 = base64.b64encode(in_data).decode("utf-8")
                # Create event
                event = {"type": "input_audio_buffer.append", "audio": audio_base64}
                # Send event as JSON string
                if self.ws is not None:
                    await self.ws.send(json.dumps(event))
                else:
                    raise websockets.ConnectionClosedError(
                        1006, "WebSocket connection is closed."
                    )
        except asyncio.CancelledError:
            logger.info("send_audio_stream task cancelled.")
        except websockets.exceptions.ConnectionClosedOK as e:
            logger.info("WebSocket connection closed gracefully: %s", e)
        except Exception as e:
            logger.exception("Unexpected error in send_audio_stream: %s", e)
        finally:
            # Ensure the audio stream is stopped
            if self.audio_manager.is_input_stream_active():
                self.audio_manager.stop_streams()

    async def display_audio_meters(self) -> None:
        """Displays audio levels in the CLI."""
        try:
            while True:
                # Retrieve levels in a thread-safe manner
                input_level, output_level = await asyncio.to_thread(
                    self.audio_manager.get_levels
                )
                self.print_audio_levels(input_level, output_level)
                await asyncio.sleep(0.2)  # Update every 100ms
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Unexpected error in display_audio_meters: %s", e)

    def print_audio_levels(self, input_level: float, output_level: float) -> None:
        """Prints the audio levels to the CLI.

        Args:
            input_level (float): Input audio RMS level.
            output_level (float): Output audio RMS level.
        """
        columns, _ = shutil.get_terminal_size(fallback=(80, 24))
        max_bar_length = (columns - 20) // 2

        def level_to_bar(level: float) -> str:
            """Converts the RMS level to a visual bar.

            Args:
                level (float): RMS level in the range [0.0, 1.0]

            Returns:
                str: Visual bar representation of the level
            """

            bar_length = int(level * max_bar_length)
            return "#" * bar_length + "-" * (max_bar_length - bar_length)

        input_bar = level_to_bar(input_level)
        output_bar = level_to_bar(output_level)
        # Note: Using print() here for real-time CLI display
        print(f"\rInput:[{input_bar}] Output:[{output_bar}]", end="", flush=True)

    async def close(self) -> None:
        """Closes the WebSocket connection and stops audio streams."""
        if self.ws is not None:
            await self.ws.close()
            logger.info("WebSocket connection closed.")
        if self.audio_manager.is_input_stream_active():
            self.audio_manager.stop_streams()
        # Signal the send_audio_stream loop to exit
        try:
            self.audio_queue.put_nowait(None)
        except Exception as e:
            logger.exception("Error putting sentinel into audio_queue: %s", e)

    async def run(self) -> None:
        """Runs the WebSocket manager to handle connection and communication."""
        try:
            await self.connect()
        except Exception as e:
            logger.exception("Unexpected error in run: %s", e)
        finally:
            await self.close()
