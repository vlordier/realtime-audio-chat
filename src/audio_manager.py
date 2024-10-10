import threading
import numpy as np
import pyaudio
import logging
import queue  # For thread-safe queues
import scipy.signal as signal

from src.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class AudioManager:
    """Handles asynchronous audio input/output using pyaudio, with AGC, compression, filtering, and noise reduction."""

    def __init__(self, config_manager: ConfigManager):
        """Initializes the AudioManager.

        Args:
            config_manager (ConfigManager): The configuration manager instance.
        """
        self.p = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.audio_queue: queue.Queue = queue.Queue()
        self.config_manager = config_manager
        self.input_level = 0.0  # RMS level of input
        self.output_level = 0.0  # RMS level of output
        self.level_lock = threading.Lock()

        self.echo_cancellation_enabled = self.config_manager.get(
            "enable_echo_cancellation"
        )  # Configurable echo cancellation
        logger.info(
            f"Echo cancellation {'enabled' if self.echo_cancellation_enabled else 'disabled'}."
        )

        # AGC parameters
        self.target_rms = 0.1  # Target RMS level for AGC
        self.agc_max_gain = 10.0
        self.agc_min_gain = 0.1
        self.current_gain = 1.0

        # Compressor parameters (dbx 160-style)
        self.compression_threshold_db = -10  # Compression threshold in dB
        self.compression_ratio = 4.0  # Compression ratio (fixed 4:1)
        self.makeup_gain_db = 6  # Makeup gain after compression

        # Noise gate and filter parameters
        self.noise_gate_threshold = 0.02  # Noise gate threshold (normalized)
        self.noise_threshold = 0.02 * 32768.0  # Noise reduction threshold

        # Adaptive filter for echo cancellation
        self.filter_length = 2 * 1024
        self.step_size = 0.001
        self.adaptive_filter = np.zeros(self.filter_length)

        # Precompute filter coefficients for passband filter
        lowcut = 300.0
        highcut = 3500.0
        sample_rate = 24000
        nyquist = 0.5 * sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        self.b, self.a = signal.butter(4, [low, high], btype="band")

    # === Main Input Processing ===
    def process_input(self, audio_data: bytes) -> bytes:
        """Processes input audio by applying noise gate, noise reduction, AGC, compression, and passband filtering."""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

            # Apply passband filter (80 Hz to 3000 Hz)
            audio_array = self.apply_passband_filter(audio_array)

            # Apply noise gate for voice detection
            audio_array = self.apply_noise_gate(audio_array)

            # Apply noise reduction
            audio_array = self.apply_noise_reduction(audio_array)

            # Apply AGC
            audio_array = self.apply_agc(audio_array)

            # Apply dbx 160-style compression
            audio_array = self.apply_compression_dbx_160(audio_array)

            return audio_array.astype(np.int16).tobytes()
        except Exception as e:
            logger.exception("Error processing input audio: %s", e)
            return audio_data

    # === Noise Gate ===
    def apply_noise_gate(self, audio_array: np.ndarray) -> np.ndarray:
        """Applies a noise gate that opens based on RMS level after passband filtering."""
        rms = self.compute_rms_level(audio_array)

        # If RMS is below the noise gate threshold, mute the audio
        if rms < self.noise_gate_threshold:
            return np.zeros_like(audio_array)
        return audio_array

    # === Noise Reduction ===
    def apply_noise_reduction(self, audio_array: np.ndarray) -> np.ndarray:
        """Applies noise reduction to filter out low-level background noise."""
        return np.where(np.abs(audio_array) < self.noise_threshold, 0, audio_array)

    # === AGC (Automatic Gain Control) ===
    def apply_agc(self, audio_array: np.ndarray) -> np.ndarray:
        """Applies Automatic Gain Control to the audio data."""
        rms = self.compute_rms_level(audio_array)
        if rms > 0:
            desired_gain = self.target_rms / rms
            self.current_gain = np.clip(
                desired_gain, self.agc_min_gain, self.agc_max_gain
            )
            audio_array *= self.current_gain
        return audio_array

    def apply_compression_dbx_160(self, audio_array: np.ndarray) -> np.ndarray:
        """Applies dbx 160-style hard knee compression with a 4:1 ratio."""
        # Remove unused threshold_linear variable
        epsilon = 1e-10  # Small value to avoid log(0)
        max_val = 32767.0  # Maximum value for 16-bit audio
        sample_db = 20 * np.log10(np.abs(audio_array) / max_val + epsilon)

        # Create a mask for samples above the threshold
        mask = sample_db > self.compression_threshold_db

        # For samples above threshold, apply compression
        excess_db = sample_db[mask] - self.compression_threshold_db
        compressed_db = self.compression_threshold_db + (
            excess_db / self.compression_ratio
        )
        compressed_linear = 10 ** (compressed_db / 20.0)
        audio_array[mask] = np.sign(audio_array[mask]) * compressed_linear * max_val

        # Apply makeup gain only to compressed samples
        makeup_gain_linear = 10 ** (self.makeup_gain_db / 20.0)
        audio_array[mask] *= makeup_gain_linear

        # Ensure the output values are within the valid range
        audio_array = np.clip(audio_array, -32768, 32767)

        return audio_array.astype(np.int16)

    # === Passband Filter ===
    def apply_passband_filter(self, audio_array: np.ndarray) -> np.ndarray:
        """Applies a passband filter for voice frequencies (80 Hz to 3000 Hz)."""
        return signal.lfilter(self.b, self.a, audio_array)

    # === RMS Calculation ===
    def compute_rms_level(self, audio_array: np.ndarray) -> float:
        """Safely computes the RMS level of the audio data.

        Args:
            audio_array (np.ndarray): The audio data to process.

        Returns:
            float: The RMS level of the audio data or 0.0 if invalid values are encountered.
        """
        try:
            # If audio_array is bytes, convert it to a NumPy array
            if isinstance(audio_array, bytes):
                audio_array = np.frombuffer(audio_array, dtype=np.int16)

            # Ensure audio_array is of float type
            audio_array = audio_array.astype(np.float32)

            # Check if the array is empty
            if len(audio_array) == 0:
                return 0.0

            # Ensure there are no NaN or Inf values
            if np.isnan(audio_array).any() or np.isinf(audio_array).any():
                logger.warning("Invalid values (NaN or Inf) found in audio data.")
                return 0.0

            # Compute the mean of the squared values
            mean_square = np.mean(np.square(audio_array))

            # Ensure the mean_square is positive and valid
            if mean_square <= 0 or np.isnan(mean_square) or np.isinf(mean_square):
                return 0.0

            # Compute the RMS value, normalized to 0.0-1.0 range
            rms = np.sqrt(mean_square) / 32768.0

            # Ensure the computed RMS is valid and within expected range
            if np.isnan(rms) or np.isinf(rms) or rms < 0.0 or rms > 1.0:
                return 0.0

            return float(rms)

        except Exception as e:
            logger.exception("Failed to compute RMS level: %s", e)
            return 0.0

    # === Echo Cancellation (LMS Adaptive Filter) ===
    def apply_echo_cancellation(self, mic_input: bytes) -> bytes:
        """Applies echo cancellation using an adaptive filter (LMS).

        Args:
            mic_input (bytes): The microphone input audio data.

        Returns:
            bytes: Echo-cancelled audio data.
        """
        try:
            # Convert mic input to a numpy array
            mic_input_array = np.frombuffer(mic_input, dtype=np.int16).astype(
                np.float32
            )

            # Simulate the speaker output (should be captured from the real output stream)
            # For this example, we'll use mic input; replace with real output
            speaker_output = mic_input_array.copy()

            # Adaptive filter processing (LMS algorithm)
            # Vectorized implementation
            num_samples = len(mic_input_array)
            if num_samples > self.filter_length:
                # Prepare reference signals
                ref_signals = np.lib.stride_tricks.sliding_window_view(
                    speaker_output, self.filter_length
                )
                mic_signals = mic_input_array[self.filter_length :]

                # Check for NaN or Inf values
                valid_indices = ~(
                    np.isnan(ref_signals).any(axis=1)
                    | np.isinf(ref_signals).any(axis=1)
                )

                ref_signals = ref_signals[valid_indices]
                mic_signals = mic_signals[valid_indices]

                # Clipping to avoid overflow
                ref_signals = np.clip(ref_signals, -1e5, 1e5)
                filtered_outputs = ref_signals @ self.adaptive_filter

                errors = mic_signals - filtered_outputs
                errors = np.clip(errors, -1e5, 1e5)

                # Update adaptive filter coefficients
                self.adaptive_filter += np.clip(
                    2 * self.step_size * (errors[:, None] * ref_signals).mean(axis=0),
                    -1e5,
                    1e5,
                )

            # Convert the echo-cancelled signal back to bytes and return it
            return mic_input_array.astype(np.int16).tobytes()

        except Exception as e:
            logger.exception("Failed to apply echo cancellation: %s", e)
            return mic_input  # Return the original input if an exception occurs

    def start_input_stream(self, callback: callable) -> None:
        """Starts the input stream using pyaudio.

        Args:
            callback (callable): The callback function to process audio data.
        """
        if self.is_input_stream_active():
            logger.info("Input stream is already active.")
            return

        try:
            input_device_name = self.config_manager.get_input_device()
            input_device_index = self.get_device_index_by_name(
                input_device_name, input=True
            )
        except Exception as e:
            logger.exception("Failed to get input device: %s", e)
            return

        try:
            self.input_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                input=True,
                input_device_index=input_device_index,
                frames_per_buffer=1024,
                stream_callback=callback,
            )
            if self.input_stream is not None:
                self.input_stream.start_stream()
            else:
                raise ValueError("Input stream is not initialized.")
            logger.info("Microphone input stream started.")
        except Exception as e:
            logger.exception("Failed to start input stream: %s", e)

    def is_output_stream_active(self) -> bool:
        """Checks if the output stream is active.

        Returns:
            bool: True if the output stream is active, False otherwise.
        """
        return self.output_stream is not None and self.output_stream.is_active()

    def is_input_stream_active(self) -> bool:
        """Checks if the input stream is active.

        Returns:
            bool: True if the input stream is active, False otherwise.
        """
        return self.input_stream is not None and self.input_stream.is_active()

    def start_output_stream(self) -> None:
        """Starts the output stream using pyaudio."""
        if self.is_output_stream_active():
            logger.info("Output stream is already active.")
            return

        try:
            output_device_name = self.config_manager.get_output_device()
            output_device_index = self.get_device_index_by_name(
                output_device_name, input=False
            )
        except Exception as e:
            logger.exception("Failed to get output device: %s", e)
            return

        try:
            self.output_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True,
                output_device_index=output_device_index,
            )
            self.output_stream.start_stream()
            logger.info("Speaker output stream started.")
        except Exception as e:
            logger.exception("Failed to start output stream: %s", e)

    def write_output(self, audio_data: bytes) -> None:
        """Writes audio data to the output stream using pyaudio.

        Args:
            audio_data (bytes): The audio data to be played.
        """
        try:
            if not self.is_output_stream_active():
                logger.warning("Output stream is not active. Starting output stream.")
                self.start_output_stream()

            # Write audio data to the output stream
            if self.output_stream is not None:
                self.output_stream.write(audio_data)
            else:
                raise ValueError("Output stream is not initialized.")

            # Compute output level using the new function to handle NaN and empty buffers
            output_rms = self.compute_rms_level(
                np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            )
            self.set_output_level(output_rms)

        except Exception as e:
            logger.exception("Failed to write output audio data: %s", e)

    def stop_streams(self) -> None:
        """Stops the input and output streams."""
        try:
            if self.input_stream is not None:
                self.input_stream.stop_stream()
                self.input_stream.close()
                self.input_stream = None
                logger.info("Microphone input stream stopped.")
        except Exception as e:
            logger.exception("Failed to stop input stream: %s", e)

        try:
            if self.output_stream is not None:
                self.output_stream.stop_stream()
                self.output_stream.close()
                self.output_stream = None
                logger.info("Speaker output stream stopped.")
        except Exception as e:
            logger.exception("Failed to stop output stream: %s", e)

        try:
            self.p.terminate()
            logger.info("pyaudio terminated.")
        except Exception as e:
            logger.exception("Failed to terminate pyaudio: %s", e)

    def get_device_index_by_name(self, device_name: str, input: bool = True) -> int:
        """Gets the device index from its name using pyaudio.

        Args:
            device_name (str): The name of the device to find.
            input (bool, optional): True to search input devices, False for output devices. Defaults to True.

        Raises:
            ValueError: If the device is not found.

        Returns:
            int: The index of the device.
        """
        device_count = self.p.get_device_count()
        for idx in range(device_count):
            device_info = self.p.get_device_info_by_index(idx)
            if device_name.lower() in device_info["name"].lower():
                if input and device_info["maxInputChannels"] > 0:
                    return idx
                elif not input and device_info["maxOutputChannels"] > 0:
                    return idx
        # Log available devices for debugging
        self.log_available_devices(input=input)
        raise ValueError(f"Device '{device_name}' not found.")

    def log_available_devices(self, input: bool = True) -> None:
        """Logs all available input or output devices.

        Args:
            input (bool, optional): True to log input devices, False for output devices. Defaults to True.
        """
        device_count = self.p.get_device_count()
        devices = []
        for idx in range(device_count):
            device_info = self.p.get_device_info_by_index(idx)
            if input and device_info["maxInputChannels"] > 0:
                devices.append(device_info["name"])
            elif not input and device_info["maxOutputChannels"] > 0:
                devices.append(device_info["name"])
        device_type = "Input" if input else "Output"
        logger.info(f"Available {device_type} Devices: {devices}")

    def set_input_level(self, level: float) -> None:
        """Sets the input audio level.

        Args:
            level (float): The RMS level of the input audio.
        """
        with self.level_lock:
            self.input_level = level

    def set_output_level(self, level: float) -> None:
        """Sets the output audio level.

        Args:
            level (float): The RMS level of the output audio.
        """
        with self.level_lock:
            self.output_level = level

    def get_levels(self) -> tuple:
        """Gets the current input and output audio levels.

        Returns:
            tuple: A tuple containing the input and output audio levels.
        """
        with self.level_lock:
            return self.input_level, self.output_level
