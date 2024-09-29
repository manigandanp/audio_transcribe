import os
import librosa
from faster_whisper import WhisperModel, BatchedInferencePipeline
from clearml import Task
import numpy as np
import soundfile as sf
import csv

class TranscriptionPipeline:
    def __init__(self, model_size="large-v3", language='ta', chunk_length=10):
        self.language = language
        self.chunk_length = chunk_length
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")
        self.batched_model = BatchedInferencePipeline(
            model=self.model, language=self.language, chunk_length=self.chunk_length
        )

    def transcribe(self, audio):
        segments, _ = self.batched_model.transcribe(audio, batch_size=16)
        return [(seg.start, seg.end, seg.text) for seg in segments]

class BatchProcessor:
    def __init__(self, input_task_id: str, output_task_id: str, batch_index: int, batch_size: int):
        self.WISHPER_TRANSCRIPTION_SAMPLING_RATE = 16000
        self.input_task = Task.get_task(task_id=input_task_id)
        self.output_task = Task.get_task(task_id=output_task_id)
        self.batch_index = batch_index
        self.batch_size = batch_size
        self.transcription_pipeline = TranscriptionPipeline()

    # def process_batches(self):
    #     artifacts = self.input_task.artifacts
    #     total_batches = (len(artifacts) + self.batch_size - 1) // self.batch_size
    #     for i in range(0, len(artifacts), self.batch_size):
    #         batch = list(artifacts.keys())[i:i+self.batch_size]
    #         print(f"Processing batch {i//self.batch_size + 1} of {total_batches}")
    #         self.process_batch(batch)
    #     print("All batches processed.")
    #     self.delete_original_artifacts()
    #     print("Closing task...")
    #     self.close_task()

    def process_batch(self):
        artifacts = self.input_task.artifacts
        all_keys = list(artifacts.keys())
        start_idx = self.batch_index * self.batch_size
        end_idx = min((self.batch_index + 1) * self.batch_size, len(all_keys))
        batch = all_keys[start_idx:end_idx]
        
        transcriptions = []
        for audio_name in batch:
            print(f"Processing started for {audio_name}")
            audio = self.input_task.artifacts[audio_name].get()
            audio_array = audio['array'].astype(np.float32)
            orig_sr = audio['sampling_rate']
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=self.WISHPER_TRANSCRIPTION_SAMPLING_RATE)
            print(f"Transcribing started for {audio_name}")
            transcription = self.transcription_pipeline.transcribe(audio_array)
            transcriptions.append((audio_name, transcription))
        
        self.chunk_and_upload(transcriptions)

    def chunk_and_upload(self, transcriptions):
        for audio_name, trans in transcriptions:
            audio = self.input_task.artifacts[audio_name].get()
            audio_name = audio_name.replace('.wav', '').strip()
            chunks_dir = f"{audio_name}-wavs"
            os.makedirs(chunks_dir, exist_ok=True)
            _, metadata = self.chunk_audio(audio, trans, audio_name, chunks_dir)
            print(f"uploading chunks... - {audio_name}")
            self.output_task.upload_artifact(name=chunks_dir, artifact_object=chunks_dir)
            metadata_path =  f"{audio_name}_metadata.csv" # os.path.join(chunks_dir, f"{audio_name}_metadata.csv")
            self.create_metadata_csv(metadata_path, metadata)
            print(f"uploading metadata... - {audio_name}")
            self.output_task.upload_artifact(f"{audio_name}_metadata", metadata_path)

    @staticmethod
    def chunk_audio(audio, transcription, audio_name, output_dir):
        chunk_names = []
        metadata = []
        audio_array = audio['array']
        sampling_rate = audio['sampling_rate']
        print(f"Chunking started for {audio_name} and {len(transcription)} - transcriptions")
        for start, end, text in transcription:
            start_sample = int(start*sampling_rate)
            end_sample = int(end*sampling_rate)
            chunk_name = f"{audio_name}-{start_sample}_{end_sample}.wav"
            chunk_path = os.path.join(output_dir, chunk_name)
            chunk = audio_array[start_sample:end_sample]
            sf.write(chunk_path, chunk, sampling_rate)
            chunk_names.append(chunk_name)
            
            duration = end - start
            metadata.append({
                'chunk_name': chunk_name,
                'transcription': text,
                'start': start,
                'end': end,
                'duration': duration,
                'original_audio': f"{audio_name}.wav"
            })
        print(f"Chunking completed for {audio_name}")
        return chunk_names, metadata

    @staticmethod
    def create_metadata_csv(file_path, metadata):
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['chunk_name', 'transcription', 'start', 'end', 'duration', 'original_audio']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in metadata:
                writer.writerow(row)
                
    def close_task(self):
        print("Closing the output task...")
        self.output_task.close()
        print("Task closed successfully.")

    def delete_original_artifacts(self):
        print("Deleting original audio artifacts...")
        artifacts = self.input_task.artifacts
        artifact_names_to_delete = [name for name in artifacts.keys() if name.endswith('.wav')]
        
        if artifact_names_to_delete:
            try:
                success = self.input_task.delete_artifacts(
                    artifact_names=artifact_names_to_delete,
                    raise_on_errors=False,
                    delete_from_storage=True,
                    silent_on_errors=False
                )
                if success:
                    print(f"Successfully deleted {len(artifact_names_to_delete)} original audio artifacts.")
                else:
                    print("Failed to delete some or all artifacts.")
            except Exception as e:
                print(f"An error occurred while deleting artifacts: {e}")
        else:
            print("No .wav artifacts found to delete.")
                
if __name__ == "__main__":
    task = Task.current_task()
    task_parameters = task.get_parameters_as_dict()["General"]
    input_task_id = task_parameters.get("input_task_id")
    output_task_id = task_parameters.get("output_task_id")
    batch_index = task_parameters.get("batch_index")
    batch_size = task_parameters.get("batch_size")
    batch_processor = BatchProcessor(input_task_id, output_task_id, batch_index, batch_size)
    batch_processor.process_batch()