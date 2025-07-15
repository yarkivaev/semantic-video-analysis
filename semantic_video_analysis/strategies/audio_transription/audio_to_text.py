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

    def transcribe_audio(self, audio_path: str) -> list[dict]:
        """
        Transcribes the audio from a file to text using the Whisper model.

        Args:
            audio_path (str): The path to the input audio file.

        Returns:
            list[dict]: List of transcription segments with start, end, and text.
                       Each dict contains: {'start': float, 'end': float, 'text': str}

        """
        result = self.model.transcribe(audio_path)
        print(result["text"], result.get("segments", []), result["language"])

        # Extract segments with timestamps
        segments = []
        for segment in result.get("segments", []):
            segments.append({
                'start': segment['start'],
                'end': segment['end'], 
                'text': segment['text'].strip()
            })
        
        return segments
    
