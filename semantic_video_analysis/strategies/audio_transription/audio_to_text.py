import whisper 

class AudioToTextTranslator:
    """
    A class to handle audio to text transcription using the Whisper model.
    """

    def __init__(self, model_name: str = "large-v3", device: str = "cuda"):
        """
        Initializes the AudioToTextTranslator with the specified model and device.

        Args:
            model_name (str): The name of the Whisper model to use.
            device (str): The device to run the model on ('cuda' or 'cpu').
        """
        self.device = device
        self.model = whisper.load_model(model_name, device=self.device)

    def transcribe_audio(self, audio_path: str) -> tuple:
        """
        Transcribes the audio from a file to text using the Whisper model.

        Args:
            audio_path (str): The path to the input audio file.

        Returns:

            - str: The transcribed text.

        """
        result = self.model.transcribe(audio_path)
        print(result["text"], result.get("segments", []), result["language"])

        return result["text"]
    
