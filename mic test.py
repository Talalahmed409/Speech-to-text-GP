import pyaudio
import wave

def list_input_devices(p):
    """List all input devices and allow user to choose one."""
    print("Available audio input devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info.get("maxInputChannels") > 0:  
            print(f"{i}: {device_info.get('name')}")

    device_index = int(input("Enter the device index of your preferred microphone: "))
    return device_index

def record_audio(device_index, output_filename="output.wav", record_seconds=10):
    """Record audio from the specified device and save it to a .wav file."""
    CHUNK_SIZE = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  
    RATE = 16000  

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE,
        input_device_index=device_index
    )

    print("Recording...")
    frames = []

    for _ in range(0, int(RATE / CHUNK_SIZE * record_seconds)):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

p = pyaudio.PyAudio()
device_index = list_input_devices(p)
p.terminate()

record_audio(device_index)
