# process_audio.py

import os
import argparse
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio
from faster_whisper import WhisperModel, BatchedInferencePipeline
from clearml import Task
from dataset_loader import get_dataset

def transcribe(audio_array, language='ta', chunk_length=10):
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    batched_model = BatchedInferencePipeline(model=model, language=language, chunk_length=chunk_length)
    segments, _ = batched_model.transcribe(audio_array, batch_size=16)
    return segments

def process_audio(project_name, hf_dataset_name, hf_config_name, audio_index, audio_name, queue_name = "process_audio"):
    # Initialize ClearML Task
    # task = Task.init(project_name=project_name, task_name=f"Process Audio - {audio_name}-{audio_index}")
    # task.add_requirements("requirements.txt")
    # task.execute_remotely(queue_name)
    # logger = task.get_logger()
    task = Task.current_task()
    token = os.environ.get("HF_TOKEN")

    # Load the specific audio file from Hugging Face dataset
    dataset = get_dataset(hf_dataset_name, hf_config_name, token)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    audio_data = dataset[audio_index]

    # Extract audio data
    audio_array = audio_data['audio']['array']
    sample_rate = audio_data['audio']['sampling_rate']
    original_filename = audio_data.get('filename', f'audio_{audio_index}.wav')

    # # Transcribe using faster-whisper
    segments = transcribe(audio_array)
    
    # Prepare directories
    os.makedirs('wavs', exist_ok=True)

    # Collect transcription and timestamps
    transcription_data = []
    for i, segment in enumerate(segments):
        start = segment.start  # in seconds
        end = segment.end      # in seconds
        text = segment.text

        # Split audio segment
        segment_audio_array = audio_array[int(start * sample_rate):int(end * sample_rate)]
        segment_filename = f'{original_filename.replace(".wav", f"_{i}.wav")}'
        segment_path = os.path.join('wavs', segment_filename)
        sf.write(segment_path, segment_audio_array, sample_rate)

        # Append to transcription data
        transcription_data.append({
            'transcription': text,
            'start_time_in_sec': start,
            'end_time_in_sec': end,
            'split_audio_name': segment_filename,
            'original_filename': original_filename,
            'dataset_name': hf_dataset_name,
            "dataset_subfolder": hf_config_name
        })

    # Save metadata.csv
    metadata_df = pd.DataFrame(transcription_data)
    metadata_csv_path = 'metadata.csv'
    metadata_df.to_csv(metadata_csv_path, index=False)

    # Upload artifacts to ClearML
    task.upload_artifact(name='metadata.csv', artifact_object=metadata_csv_path)
    task.upload_artifact(name='wavs', artifact_object='wavs')

    # Finalize task
    task.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, required=True)
    parser.add_argument('--hf_dataset_name', type=str, required=True)
    parser.add_argument('--hf_config_name', type=str, required=True)
    parser.add_argument('--audio_index', type=int, required=True)
    parser.add_argument('--audio_name', type=str, required=True)
    args = parser.parse_args()

    process_audio(args.project_name, args.hf_dataset_name, args.hf_config_name, args.audio_index, args.audio_name)
