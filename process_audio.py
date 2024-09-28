import os
import argparse
import pandas as pd
import numpy as np
import soundfile as sf
from datasets import Audio
from faster_whisper import WhisperModel, BatchedInferencePipeline
from clearml import Task
from dataset_loader import get_dataset
from abc import ABC, abstractmethod


class AudioProcessor(ABC):
    @abstractmethod
    def process(self, audio_array, sample_rate):
        pass


class TranscriptionProcessor(AudioProcessor):
    def __init__(self, language="ta", chunk_length=10):
        self.language = language
        self.chunk_length = chunk_length  # in seccods
        self.model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        self.batched_model = BatchedInferencePipeline(
            model=self.model, language=self.language, chunk_length=self.chunk_length
        )

    def process(self, audio_array, sample_rate):
        segments, _ = self.batched_model.transcribe(audio_array, batch_size=16)
        return segments


class TranscriptionFormatter:
    def __init__(self, hf_dataset_name, hf_config_name):
        self.hf_dataset_name = hf_dataset_name
        self.hf_config_name = hf_config_name

    def format(self, audio_array, sample_rate, original_filename, segments):
        transcription_data = []
        for i, segment in enumerate(segments):
            start = segment.start
            end = segment.end
            text = segment.text

            segment_audio_array = audio_array[
                int(start * sample_rate) : int(end * sample_rate)
            ]
            segment_filename = f'{original_filename.replace(".wav", f"_{i}.wav")}'
            segment_path = os.path.join("wavs", segment_filename)
            sf.write(segment_path, segment_audio_array, sample_rate)

            transcription_data.append(
                {
                    "transcription": text,
                    "start_time_in_sec": start,
                    "end_time_in_sec": end,
                    "audio_path": segment_filename,
                    "original_filename": original_filename,
                    "dataset_name": self.hf_dataset_name,
                    "dataset_subfolder": self.hf_config_name,
                }
            )

        return transcription_data


class AudioDataHandler:
    def __init__(self, hf_dataset_name, hf_config_name, token):
        self.hf_dataset_name = hf_dataset_name
        self.hf_config_name = hf_config_name
        self.token = token
        self.whisper_model_sample_rate = 16000

    def get_audio_data(self, audio_index):
        original_dataset = get_dataset(
            self.hf_dataset_name, self.hf_config_name, self.token
        )
        dataset = original_dataset.cast_column(
            "audio", Audio(sampling_rate=self.whisper_model_sample_rate)
        )
        audio_data = dataset[audio_index]

        original_audio_array = original_dataset["audio"]["array"]
        original_sample_rate = original_dataset["audio"]["sampling_rate"]

        audio_array = audio_data["audio"]["array"].astype(np.float32)
        sample_rate = audio_data["audio"]["sampling_rate"]
        original_filename = audio_data.get("filename", f"audio_{audio_index}.wav")

        return (
            original_audio_array,
            original_sample_rate,
            audio_array,
            sample_rate,
            original_filename,
        )


class ClearMLHandler:
    @staticmethod
    def upload_artifacts(task, metadata_csv_path):
        task.upload_artifact(name="metadata.csv", artifact_object=metadata_csv_path)
        task.upload_artifact(name="wavs", artifact_object="wavs")
        task.close()


class AudioProcessingPipeline:
    def __init__(
        self, project_name, hf_dataset_name, hf_config_name, audio_index, audio_name
    ):
        self.project_name = project_name
        self.hf_dataset_name = hf_dataset_name
        self.hf_config_name = hf_config_name
        self.audio_index = audio_index
        self.audio_name = audio_name
        self.task = Task.current_task()  # TODO: change this to parent task
        self.token = os.environ.get("HF_TOKEN")

    def run(self):
        audio_handler = AudioDataHandler(
            self.hf_dataset_name, self.hf_config_name, self.token
        )
        (
            original_audio_array,
            original_sample_rate,
            audio_array,
            sample_rate,
            original_filename,
        ) = audio_handler.get_audio_data(self.audio_index)

        processor = TranscriptionProcessor()
        segments = processor.process(audio_array, sample_rate)

        os.makedirs("wavs", exist_ok=True)

        formatter = TranscriptionFormatter(self.hf_dataset_name, self.hf_config_name)
        transcription_data = formatter.format(
            original_audio_array, original_sample_rate, original_filename, segments
        )

        metadata_df = pd.DataFrame(transcription_data)
        metadata_csv_path = "metadata.csv"
        metadata_df.to_csv(metadata_csv_path, index=False)

        ClearMLHandler.upload_artifacts(self.task, metadata_csv_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--hf_dataset_name", type=str, required=True)
    parser.add_argument("--hf_config_name", type=str, required=True)
    parser.add_argument("--audio_index", type=int, required=True)
    parser.add_argument("--audio_name", type=str, required=True)
    args = parser.parse_args()

    pipeline = AudioProcessingPipeline(
        args.project_name,
        args.hf_dataset_name,
        args.hf_config_name,
        args.audio_index,
        args.audio_name,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
