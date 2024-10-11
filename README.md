# Audio Transcription using Faster-Whisper (Large-v3)

## Project Overview
This project is designed to transcribe audio files using the Faster-Whisper model (large-v3). The default assumption is that the audio files are in Tamil. The project uses **ClearML** for the orchestration of the entire flow.

### Key Components:
1. **Pipeline Controller**: 
   - Takes the Huggingface dataset and its configuration name (sub-directory) as input.
   - Runs in a queue (`pipeline_worker`).
   - Creates the **Dataset Loader** task which downloads the dataset from Huggingface and uploads it to ClearML as artifacts for easier access by other tasks.
   
2. **Batch Controller Task**: 
   - Retrieves metadata about the Huggingface dataset.
   - Dynamically creates transcription tasks to process the audio files. These transcription tasks are queued into `gpu_workers`.

3. **Uploader Task**:
   - After all transcription tasks are complete, it merges the transcribed audio chunks and their metadata CSVs.
   - Uploads the merged files to Huggingface.
   - Runs on the `cpu_worker` queue.

### Queues Required:
1. **pipeline_worker**: Handles the **Pipeline Controller** (the parent of all tasks).
2. **cpu_worker**: Handles all child tasks except the **Transcription Tasks**.
3. **gpu_worker**: Handles the actual **Transcription Tasks**.

## Run Instructions

### Prerequisites:
- Ensure you have installed all dependencies from `requirements.txt`.
- Set up environment variables for ClearML and Huggingface access:
  - **ClearML**: Obtain API credentials from ClearML UI and set them as environment variables.
  - **Huggingface**: Set API credentials to access and upload datasets (especially for private datasets).

### Steps:
1. **Commit and push all changes** to the `main` branch.
2. Run the base tasks registration:
   ```bash
   python register_base_tasks.py
   ```
3. Create the pipeline:
   ```bash
   python main.py
   ```
   - After creating the pipeline, you need to **manually enqueue** the pipeline into the `pipeline_worker` queue to start the process.
   
4. **Start the workers**:
   - For GPU workers (transcription tasks):
     ```bash
     clearml-agent daemon --queue gpu_worker --foreground --create-queue
     ```
   
5. **Optional**: If you need to use multiple Google Colab instances as ClearML workers, start multiple Chrome instances with different sessions:
   ```bash
   open -na "Google Chrome" --args --user-data-dir="/tmp/chrome-session-$(date +%s)" https://www.gmail.com
   ```

## Notes:
- Ensure that the environment is properly configured with ClearML and Huggingface credentials.
- This project involves multiple tasks and dynamic orchestration via ClearML queues, so monitoring the tasks through the ClearML UI is essential for successful execution.

