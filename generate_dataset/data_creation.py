import cv2
import ffmpeg
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from tqdm import tqdm

# Input folder containing videos
videos_folder = "input_videos"
output_folder_ad = r"D:\ad_data\output_folder_ad"
output_folder_no_ad = r"D:\ad_data\output_folder_no_ad"

os.makedirs(output_folder_ad, exist_ok=True)
os.makedirs(output_folder_no_ad, exist_ok=True)

# Set chunk duration in seconds
chunk_duration = 2

# Process each video in the folder
for video_file in tqdm(os.listdir(videos_folder)):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(videos_folder, video_file)

        # Use ffmpeg-python to extract audio
        audio_output_path = os.path.join(output_folder_no_ad if 'intermediate' in video_file else output_folder_ad,
                                         f"{video_file[:-4]}_audio.wav")
        audio = (
            ffmpeg.input(video_path)
            .output(audio_output_path, acodec='pcm_s16le', ar=36000, ac=1)
            .overwrite_output()
            .run()
        )

        # Save spectrogram image
        audio_data, _ = ffmpeg.input(audio_output_path).output('-', format='wav').run(quiet=True, capture_stdout=True)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        i = 0

        # Use OpenCV to segment the video
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Calculate number of chunks
        num_chunks = int(duration / chunk_duration)

        # Read and save video chunks
        # Read and save reduced frames (15 frames per second, spaced every 6 frames) as numpy array
        video_frames = []
        for chunk_num in range(num_chunks):
            frames = []
            for _ in range(int(chunk_duration * fps)):
                ret, frame = cap.read()
                if not ret:
                    break
                if _ % 12 == 0:
                    frames.append(cv2.resize(frame, (320, 320)))  # Resize frames to (640, 640)
            # Save video frames as numpy array
            video_frames_array_path = os.path.join(output_folder_no_ad if 'intermediate' in video_file else output_folder_ad,
                                                   f"{video_file[:-4]}_video_frames_chunk_{chunk_num+1}.npy")
            np.save(video_frames_array_path, np.array(frames))

            fs = 36000
            frequencies, times, Sxx = spectrogram(audio_array[fs*i*1:fs*(i+1)*1], fs=fs, nperseg=1024, noverlap=128)
            i = i + 1
            plt.figure(figsize=(10, 6))
            plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto', cmap='viridis')
            plt.axis('off')
            plt.gca().set_position([0, 0, 1, 1])
            plt.savefig(os.path.join(output_folder_no_ad if 'intermediate' in video_file else output_folder_ad,
                                     f"{video_file[:-4]}_spectrogram_chunk_{chunk_num+1}.png"),
                        bbox_inches='tight', pad_inches=0)
            plt.close()

        # Release OpenCV resources for the current video
        cap.release()
        # Remove the audio file after processing
        os.remove(audio_output_path)
