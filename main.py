""" Main entry point for the Audio Streamer for OpenAI GPT-4o Realtime API """

import asyncio
import logging
import argparse

from src.audio_streamer import AudioStreamer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for Windows event loop policy
# import sys
# if sys.platform.startswith("win") and sys.version_info >= (3, 8):
#     asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio Streamer for OpenAI GPT-4o Realtime API"
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update the configuration (e.g., select audio devices)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="audio_config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode for detailed logging"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")

    audio_streamer = AudioStreamer(
        config_file=args.config_file, update_config=args.update_config
    )
    asyncio.run(audio_streamer.run())
