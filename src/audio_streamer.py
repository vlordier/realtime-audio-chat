"""Main module for the audio streaming application."""

import logging
import asyncio

from src.config_manager import ConfigManager
from src.audio_manager import AudioManager
from src.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


class AudioStreamer:
    """Manages the audio streaming, integrating audio and WebSocket management."""

    def __init__(self, config_file: str = None, update_config: bool = False) -> None:
        """Initializes the AudioStreamer with configuration, audio, and WebSocket managers.

        Args:
            config_file (str, optional): Path to the configuration file. Defaults to None.
            update_config (bool, optional): Whether to update the configuration. Defaults to False.
        """
        self.config_manager = ConfigManager(
            config_file=config_file, update_config=update_config
        )
        self.audio_manager = AudioManager(self.config_manager)
        self.ws_manager = WebSocketManager(self.config_manager, self.audio_manager)
        logger.info("AudioStreamer initialized.")

    async def run(self) -> None:
        """Main run method that initiates connections and handles audio streaming."""
        try:
            await self.ws_manager.run()
        except asyncio.CancelledError:
            logger.info("AudioStreamer run cancelled.")
            raise  # Re-raise to allow proper cancellation handling
        except Exception as e:
            logger.exception("Application error: %s", e)
        finally:
            # Ensure streams are stopped if active
            if self.audio_manager.is_input_stream_active():
                self.audio_manager.stop_streams()
            logger.info("AudioStreamer stopped.")
