import os
import subprocess
import time
from threading import Thread

import cv2
import numpy as np
import torch
from flask import Flask
from gevent.pywsgi import WSGIServer
from melo.api import TTS
from openvoice.api import ToneColorConverter

import audio
from models import Wav2Lip

device = "cuda:0" if torch.cuda.is_available() else "cpu"
video_pipe_name = "video_pipe"
audio_pipe_name = "audio_pipe"
fd_v: int
fd_a: int
v_full_idle: np.ndarray
a_full_idle: np.ndarray
access = True


def datagen(frames, mels, face_det_results, config: dict):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    for i, m in enumerate(mels):
        idx = 0 if config['static'] else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (config['img_size'], config['img_size']))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= config['wav2lip_batch_size']:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch.copy()
            img_masked[:, config['img_size'] // 2:] = 0
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
        img_masked = img_batch.copy()
        img_masked[:, config['img_size'] // 2:] = 0
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])
        yield img_batch, mel_batch, frame_batch, coords_batch


def ffmpeg_pre_process(command: list, fps: int):
    make_pipe()
    _ = run_ffmpeg(command)
    global fd_v, fd_a, video_pipe_name, audio_pipe_name, v_full_idle, a_full_idle
    Thread(target=write_idle, args=(fps, fd_v, video_pipe_name, v_full_idle)).start()
    Thread(target=write_idle, args=(fps, fd_v, audio_pipe_name, a_full_idle)).start()


def write_idle(fps: int, fd: int, pipe_name: str, full_idle: np.ndarray):
    global access
    if fd is None:
        fd = os.open(pipe_name, os.O_APPEND)
    while True:
        if access is False:
            time.sleep(0.1)
        else:
            t0 = time.time()
            for _ in range(fps):
                if access is False:
                    break
                os.write(fd, full_idle.tobytes())
            try:
                time.sleep(1 - time.time() + t0)
            except ValueError:
                ...


def make_pipe():
    # 判断如果管道存在，则先unlink
    if os.path.exists(video_pipe_name):
        os.unlink(video_pipe_name)
    if os.path.exists(audio_pipe_name):
        os.unlink(audio_pipe_name)
    os.mkfifo(video_pipe_name)
    os.mkfifo(audio_pipe_name)


def run_ffmpeg(command: list) -> subprocess.Popen:
    proc = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)
    # print("ffmpeg exit, exit code = %d" % (proc.wait()))
    return proc


def write_audio(wav_array: np.ndarray, fps: int):
    speech_array = (wav_array * 32767).astype(np.int16)
    global fd_a
    wav_frame_num = int(44100 / fps)
    frame_counter = 0
    while True:
        # 由于音频流的采样率是xxx, 而视频流的帧率是25, 因此需要对音频流进行分帧
        speech = speech_array[
                 frame_counter * wav_frame_num: (frame_counter + 1) * wav_frame_num
                 ]
        os.write(fd_a, speech.tobytes())
        frame_counter += 1
        if frame_counter * wav_frame_num >= len(speech_array):
            break


class Web:
    def __init__(self, converter: ToneColorConverter, tts_model: TTS, source_se: torch.Tensor, target_se: torch.Tensor,
                 wav2lip_model: Wav2Lip, full_frames: list, face_det_results: list, config: dict):
        self.server = None
        self.converter = converter
        self.tts_model = tts_model
        self.source_se = source_se
        self.target_se = target_se
        self.wav2lip_model = wav2lip_model
        self.full_frames = full_frames
        self.face_det_results = face_det_results
        self.frame_h, self.frame_w = full_frames[0].shape[:-1]
        self.command = [
            "ffmpeg",
            "-loglevel",
            "info",
            "-y",
            "-an",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "{}x{}".format(self.frame_w, self.frame_h),
            "-r",
            str(config['wav2lip']['fps']),
            "-i",
            video_pipe_name,  # 视频流管道作为输入
            "-i",
            audio_pipe_name,  # 音频流管道作为输入
            "-c:v",
            "libx264",
            "-s",
            "{}x{}".format(self.frame_w, self.frame_h),
            "-r",
            str(config['wav2lip']['fps']),
            "-ac",
            "1",
            "-ar",
            "44100",
            "-f",
            "rtsp",
            config['web']['push_url'],
        ]
        global v_full_idle, a_full_idle
        v_full_idle = self.full_frames[0]
        a_full_idle = (torch.full((int(44100 / config['wav2lip']['fps']),), 0)
                       .detach().cpu().numpy().astype(np.int16))
        self.config = config

    def start(self):
        app = Flask(__name__)
        speaker_ids = self.tts_model.hps.data.spk2id
        speaker_key = list(speaker_ids.keys())[0]
        speaker_id = speaker_ids[speaker_key]
        speaker_key = speaker_key.lower().replace("_", "-")
        tts_config = self.config['tts']
        wav2lip_config = self.config['wav2lip']
        src_path = f"{tts_config['output_dir']}/tmp.wav"
        wav_save_path = f"{tts_config['output_dir']}/output_v2_{speaker_key}.wav"
        mel_step_size = 16
        ffmpeg_pre_process(self.command, wav2lip_config['fps'])

        @app.route('/')
        def hello_world():
            return 'Hello, World!'

        @app.route('/text/<text>')
        def text2lip(text):
            self.tts_model.tts_to_file(text, speaker_id, src_path,
                                       speed=tts_config['speed'])
            self.converter.convert(
                audio_src_path=src_path,
                src_se=self.source_se,
                tgt_se=self.target_se,
                output_path=wav_save_path
            )
            wav = audio.load_wav(wav_save_path, 44100)
            mel = audio.melspectrogram(wav)
            print(mel.shape)
            if np.isnan(mel.reshape(-1)).sum() > 0:
                raise ValueError(
                    'Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
            mel_chunks = []
            mel_idx_multiplier = 80. / wav2lip_config['fps']
            i = 0
            while 1:
                start_idx = int(i * mel_idx_multiplier)
                if start_idx + mel_step_size > len(mel[0]):
                    mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                    break
                mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
                i += 1
            print("Length of mel chunks: {}".format(len(mel_chunks)))

            if wav2lip_config['box'][0] == -1 and wav2lip_config['static'] is True:
                face_det_results_seg = self.face_det_results[0]
            else:
                face_det_results_seg = self.face_det_results[:len(mel_chunks)]
            gen = datagen(self.full_frames, mel_chunks, face_det_results_seg, wav2lip_config)
            global fd_v, fd_a, access
            access = False
            Thread(target=write_audio, args=(wav, wav2lip_config['fps'])).start()
            for (img_batch, mel_batch, frames, coords) in gen:
                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
                with torch.no_grad():
                    pred = self.wav2lip_model(mel_batch, img_batch)
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

                for p, f, c in zip(pred, frames, coords):
                    y1, y2, x1, x2 = c
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    f[y1:y2, x1:x2] = p
                    os.write(fd_v, f.tobytes())
            access = True
            return "okay!"

        self.server = WSGIServer(('', self.config['web']['port']), app)
        self.server.serve_forever()
