""" Manages configuration parameters for the audio streamer. """

import os
import uuid
import yaml
import logging
import sounddevice as sd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigManager:
    """Handles loading and managing configurations."""

    def __init__(
        self, config_file: str = "audio_config.yaml", update_config=False
    ) -> None:
        """Initializes the ConfigManager with the config file path."""
        load_dotenv()  # Load from .env if available
        self.config_file = config_file
        self.update_config = update_config
        self.config_data = self.load_config()
        self.validate_config()
        self.machine_id = str(uuid.getnode())
        self.check_devices()

    def load_config(self) -> dict:
        """Loads the configuration from a YAML file or environment variables."""
        config = {}

        # Load from YAML config file if it exists
        if os.path.exists(self.config_file):
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        # Override with .env variables if they exist
        config["api_key"] = os.getenv("OPENAI_API_KEY")
        config["base_url"] = os.getenv(
            "BASE_URL", config.get("base_url", "wss://api.openai.com/v1/realtime")
        )
        config["model_type"] = os.getenv(
            "MODEL_TYPE", config.get("model_type", "gpt-4o-realtime-preview-2024-10-01")
        )
        config["enable_echo_cancellation"] = os.getenv(
            "ENABLE_ECHO_CANCELLATION", config.get("enable_echo_cancellation", False)
        )
        return config

    def validate_config(self):
        """Validates the configuration parameters."""
        if not self.get("api_key"):
            raise ValueError("API Key is required. Provide it via .env or config file.")

    def get(self, key: str, default=None):
        """Retrieve configuration value by key."""
        return self.config_data.get(key, default)

    def set(self, key: str, value) -> None:
        """Set a configuration value."""
        self.config_data[key] = value

    def save_config(self):
        """Saves the current configuration to the config file."""
        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.dump(self.config_data, f)
        logger.info(f"Configuration saved to {self.config_file}")

    def check_devices(self):
        """Checks if the default devices are available, and prompts the user to select devices if necessary."""
        current_machine_id = self.machine_id
        stored_machine_id = self.get("machine_id")
        stored_input_device = self.get("input_device")
        stored_output_device = self.get("output_device")

        devices_changed = False

        if stored_machine_id != current_machine_id or self.update_config:
            devices_changed = True

        # Get list of available devices
        devices = sd.query_devices()
        input_devices = [d for d in devices if d["max_input_channels"] > 0]
        output_devices = [d for d in devices if d["max_output_channels"] > 0]

        if (
            devices_changed
            or not self.device_exists(stored_input_device, input_devices)
            or not self.device_exists(stored_output_device, output_devices)
        ):
            # Prompt user to select devices
            logger.info(
                "Audio devices have changed or not set. Please select input and output devices."
            )
            input_device = self.select_device(input_devices, "input")
            output_device = self.select_device(output_devices, "output")
            self.set("input_device", input_device)
            self.set("output_device", output_device)
            self.set("machine_id", current_machine_id)
            self.save_config()
        else:
            logger.info(f"Using stored input device: {stored_input_device}")
            logger.info(f"Using stored output device: {stored_output_device}")

    def device_exists(self, device_name, device_list):
        """Checks if a device name exists in the provided device list."""
        for device in device_list:
            if device["name"] == device_name:
                return True
        return False

    def select_device(self, devices, device_type):
        """Prompts the user to select a device from the list."""
        print(f"Available {device_type} devices:")
        for idx, device in enumerate(devices):
            print(f"{idx}: {device['name']}")
        while True:
            try:
                selection = int(input(f"Select {device_type} device by index: "))
                if 0 <= selection < len(devices):
                    selected_device = devices[selection]["name"]
                    logger.info(f"Selected {device_type} device: {selected_device}")
                    return selected_device
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def get_input_device(self):
        return self.get("input_device")

    def get_output_device(self):
        return self.get("output_device")
