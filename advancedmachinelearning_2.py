import torch
import torchaudio
import IPython
import matplotlib.pyplot as plt
from torchaudio.utils import download_asset

# Type aliases for clarity
ResampledWaveform = torch.Tensor
AudioFeatures = torch.Tensor
EmissionsTensor = torch.Tensor


class TranscriptGenerator:
    """
    A class to handle speech recognition tasks using a pre-trained Wav2Vec2 model.

    This class includes methods for audio resampling, feature extraction, classification,
    and visualization of the audio processing pipeline.

    Attributes:
        device (torch.device): The device to run computations on (CPU or CUDA).
        bundle (torchaudio.pipelines): The pre-trained Wav2Vec2 model bundle.
        sample_rate (int): The sample rate expected by the model.
        labels (List[str]): The label set used for decoding emissions.
        model (torch.nn.Module): The Wav2Vec2 ASR model.
    """

    def __init__(self):
        """Initializes the TranscriptGenerator class and sets up the Wav2Vec2 model."""
        torch.random.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.sample_rate = self.bundle.sample_rate
        self.labels = self.bundle.get_labels()
        self.model = self.bundle.get_model().to(self.device)

    def display_audio_file_ipy(self, path_to_file: str) -> None:
        """
        Displays an audio file inline in a Jupyter notebook.

        Args:
            path_to_file (str): Path to the audio file.
        """
        IPython.display.Audio(path_to_file)

    def correct_sampling(self, path_to_file: str) -> ResampledWaveform:
        """
        Loads an audio file and resamples it to match the model's expected sample rate.

        Args:
            path_to_file (str): Path to the audio file.

        Returns:
            ResampledWaveform: The resampled audio waveform tensor.
        """
        waveform, sample_rate = torchaudio.load(path_to_file)
        waveform = waveform.to(self.device)

        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        return waveform

    def extract_features(self, waveform: ResampledWaveform) -> AudioFeatures:
        """
        Extracts audio features using the Wav2Vec2 model.

        Args:
            waveform (ResampledWaveform): The resampled audio waveform.

        Returns:
            AudioFeatures: The extracted features from each transformer layer.
        """
        with torch.inference_mode():
            features, _ = self.model.extract_features(waveform)
        return features

    def visualise_features(self, features: AudioFeatures) -> None:
        """
        Visualizes audio features extracted by the Wav2Vec2 model.

        Args:
            features (AudioFeatures): The extracted audio features to visualize.
        """
        fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
        for i, feats in enumerate(features):
            ax[i].imshow(feats[0].cpu(), interpolation="nearest")
            ax[i].set_title(f"Feature from transformer layer {i+1}")
            ax[i].set_xlabel("Feature dimension")
            ax[i].set_ylabel("Frame (time-axis)")
        fig.tight_layout()

    def classification(self, waveform: ResampledWaveform) -> EmissionsTensor:
        """
        Performs classification to generate emission logits from the Wav2Vec2 model.

        Args:
            waveform (ResampledWaveform): The resampled audio waveform.

        Returns:
            EmissionsTensor: The emission logits produced by the model.
        """
        with torch.inference_mode():
            emission, _ = self.model(waveform)
        return emission

    def visualise_emissions(self, emissions: EmissionsTensor) -> None:
        """
        Visualizes emission logits produced by the Wav2Vec2 model.

        Args:
            emissions (EmissionsTensor): The emission logits to visualize.
        """
        plt.imshow(emissions[0].cpu().T, interpolation="nearest")
        plt.title("Classification result")
        plt.xlabel("Frame (time-axis)")
        plt.ylabel("Class")
        plt.tight_layout()
        print("Class labels:", self.labels)

    def pipeline(self, file_path: str) -> None:
        """
        Runs the full pipeline: displaying audio, resampling, extracting features,
        visualizing features, performing classification, and visualizing emissions.

        Args:
            file_path (str): Path to the audio file.
        """
        self.display_audio_file_ipy(file_path)
        corrected = self.correct_sampling(file_path)
        features = self.extract_features(corrected)
        self.visualise_features(features)
        emissions = self.classification(corrected)
        self.visualise_emissions(emissions)


# Example usage:
if __name__ == "__main__":
    print(torch.__version__)
    print(torchaudio.__version__)

    tg = TranscriptGenerator()

    SPEECH_FILE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
    tg.pipeline(SPEECH_FILE)
