import wave
import pyaudio
import threading
import os
from faster_whisper import WhisperModel

NEON_GREEN = "\033[92m"  
RESET_COLOR = "\033[0m"  

recording = False

def list_input_devices(p):
    """List all input devices and allow user to choose one."""
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info.get("maxInputChannels") > 0: 
            print(f"{i}: {device_info.get('name')}")

    device_index = int(input("Enter the device index of your preferred microphone: "))
    return device_index

def record_audio(p, device_index, output_file):
    """Continuously record audio until the recording flag is turned off."""
    global recording
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1, 
        rate=16000,
        input=True,
        frames_per_buffer=1024,
        input_device_index=device_index
    )

    frames = []
    print("Press Enter to stop recording...\n")

    while recording:
        data = stream.read(1024)
        frames.append(data)

    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
    
    print("Recording stopped and saved to:", output_file)
    stream.stop_stream()
    stream.close()

def transcribe_audio(model, file_path):
    """Transcribe audio from the specified file."""
    segments, info = model.transcribe(file_path, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300))
    transcription = " ".join([segment.text for segment in segments])
    return transcription

def initialize_model():
    model_size = "medium"
    return WhisperModel(model_size, device="cuda", compute_type="float16")

if __name__ == "__main__":
    p = pyaudio.PyAudio()
    try:
        device_index = list_input_devices(p)

        model = initialize_model()

        output_file = "continuous_recording.wav"

        recording = True
        recording_thread = threading.Thread(target=record_audio, args=(p, device_index, output_file))
        recording_thread.start()

        input("Recording...")
        recording = False
        recording_thread.join()

        transcription = transcribe_audio(model, output_file)
        print(NEON_GREEN + transcription + RESET_COLOR)

        with open("log.txt", "w") as log_file:
            log_file.write(transcription)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        p.terminate()
