
import wave
import pyaudio
import os
from faster_whisper import WhisperModel

NEON_GREEN = "\033[92m"  # Color for printing transcription (optional)
RESET_COLOR = "\033[0m"  # Reset color

def list_input_devices(p):
    """List all input devices and allow user to choose one."""
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info.get("maxInputChannels") > 0:  # Only list input devices
            print(f"{i}: {device_info.get('name')}")

    device_index = int(input("Enter the device index of your preferred microphone: "))
    return device_index

def record_chunk(p, stream, chunk_file, main_audio, previous_frames=None):
    # Define the number of frames to record
    CHUNK_SIZE = 1024
    RECORD_SECONDS = 3  # Reduced duration for each chunk
    OVERLAP_SECONDS = 0.35  # Overlap to capture boundary words

    # Calculate frames to read for each chunk and overlap
    frames_per_chunk = int(16000 / CHUNK_SIZE * RECORD_SECONDS)
    frames_overlap = int(16000 / CHUNK_SIZE * OVERLAP_SECONDS)

    # Start with any remaining frames from the last chunk's overlap
    frames = previous_frames if previous_frames else []

    # Record new audio data
    for _ in range(frames_per_chunk):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)

    # Save the recorded frames as a .wav file for transcription
    with wave.open(chunk_file, 'wb') as wf:
        wf.setnchannels(1)  # Mono channel
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))  # Correct function call
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    # Append the recorded frames to the main audio file
    main_audio.writeframes(b''.join(frames))

    # Keep only the last frames for overlap in the next chunk
    return frames[-frames_overlap:]

def transcribe_chunk(model, file_path):
    segments, info = model.transcribe(file_path, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300))
    return " ".join([segment.text for segment in segments])

def initialize_model():
    model_size = "medium.en"
    return WhisperModel(model_size, device="cuda", compute_type="float16")

p = pyaudio.PyAudio()
stream = None
main_audio = None

try:
    # List devices and select input device
    device_index = list_input_devices(p)

    # Initialize the Whisper model
    model = initialize_model()

    # Open the main audio file for saving the entire session
    main_audio = wave.open("entire_session.wav", 'wb')
    main_audio.setnchannels(1)
    main_audio.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    main_audio.setframerate(16000)

    # Open the audio stream with the chosen device
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,  # Set to 1 for mono; adjust if you need stereo
        rate=16000,
        input=True,
        frames_per_buffer=1024,
        input_device_index=device_index
    )
    
    accumulated_transcription = ""
    previous_frames = []
    
    while True:
        chunk_file = "temp_chunk.wav"
        # Record chunk and get frames for the next chunk's overlap
        previous_frames = record_chunk(p, stream, chunk_file, main_audio, previous_frames)
        transcription = transcribe_chunk(model, chunk_file)
        print(NEON_GREEN + transcription + RESET_COLOR)
        os.remove(chunk_file)
        
        accumulated_transcription += transcription + " "  # Add space for readability

except KeyboardInterrupt:
    print("Stopping...")
    with open("log.txt", "w") as log_file:
        log_file.write(accumulated_transcription)

except Exception as e:
    print(f"Error: {e}")

finally:
    if main_audio is not None:
        main_audio.close()
    if stream is not None:  # Check if stream was successfully created
        stream.stop_stream()
        stream.close()
    p.terminate()

if __name__ == "__main__":
    model = initialize_model()