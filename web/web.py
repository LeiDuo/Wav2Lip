import base64
import os
import platform
import subprocess
import time
import json
from threading import Thread

import cv2
import numpy as np
import torch
from flask import Flask
from gevent.pywsgi import WSGIServer
from melo.api import TTS
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from openvoice.api import ToneColorConverter

from models import Wav2Lip
from web import tools

device = "cuda:0" if torch.cuda.is_available() else "cpu"
video_pipe_name = "video_pipe"
audio_pipe_name = "audio_pipe"
fd_v = -1
fd_a = -1
v_full_idle: np.ndarray
a_full_idle: np.ndarray
access = True
sync_difference = 0


def ffmpeg_pre_process(command: list, fps: int):
    make_pipe()
    _ = run_ffmpeg(command)
    Thread(target=write_video_idle, args=(fps,)).start()
    Thread(target=write_audio_idle, args=(fps,)).start()


# noinspection DuplicatedCode
def write_video_idle(fps: int):
    global fd_v, access, sync_difference
    if fd_v == -1:
        fd_v = os.open(video_pipe_name, os.O_WRONLY)
    while True:
        if access is False:
            time.sleep(1 / fps)
        else:
            t0 = time.time()
            for _ in range(fps):
                if access is False:
                    break
                sync_difference += 1
                os.write(fd_v, v_full_idle.tobytes())
            try:
                print(sync_difference)
                time.sleep(1 - time.time() + t0)
            except ValueError:
                ...


# noinspection DuplicatedCode
def write_audio_idle(fps: int):
    global fd_a, access, sync_difference
    if fd_a == -1:
        fd_a = os.open(audio_pipe_name, os.O_WRONLY)
    while True:
        if access is False:
            time.sleep(1 / fps)
        else:
            t0 = time.time()
            for _ in range(fps):
                if access is False:
                    break
                sync_difference -= 1
                os.write(fd_a, a_full_idle.tobytes())
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


def batch_eval(img_batch, mel_batch, wav2lip_model: Wav2Lip):
    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
    with torch.no_grad():
        pred = wav2lip_model(mel_batch, img_batch)
    return pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.


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

        @app.route('/')
        def hello_world():
            return 'Hello, World!'

        @app.route('/start_live')
        def start_live():
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
                # "-r",
                # str(config['wav2lip']['fps']),
                "-i",
                video_pipe_name,  # 视频流管道作为输入
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "-i",
                audio_pipe_name,  # 音频流管道作为输入
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-s",
                "{}x{}".format(self.frame_w, self.frame_h),
                "-r",
                str(self.config['wav2lip']['fps']),
                "-c:a",
                "aac",
                # "-ar",
                # "44100",
                "-f",
                "rtsp",
                self.config['web']['push_url'],
            ]
            global v_full_idle, a_full_idle
            v_full_idle = self.full_frames[0]
            a_full_idle = (torch.full((int(44100 / self.config['wav2lip']['fps']),), 0)
                           .detach().cpu().numpy().astype(np.int16))
            ffmpeg_pre_process(self.command, int(wav2lip_config['fps']))

        @app.route('/text_live/<text>')
        def text2lip_live(text):
            tools.text2wav(self.tts_model, self.converter, text, tts_config, speaker_id, src_path, self.source_se,
                           self.target_se, wav_save_path)
            wav, gen = tools.wav2lip_pre(wav_save_path, wav2lip_config, mel_step_size, self.full_frames,
                                         self.face_det_results)
            global fd_v, fd_a, access
            access = False
            Thread(target=write_audio, args=(wav, wav2lip_config['fps'])).start()
            for (img_batch, mel_batch, frames, coords) in gen:
                pred = batch_eval(img_batch, mel_batch, self.wav2lip_model)

                for p, f, c in zip(pred, frames, coords):
                    y1, y2, x1, x2 = c
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    f[y1:y2, x1:x2] = p
                    os.write(fd_v, f.tobytes())
            access = True
            return "okay!"

        @app.route('/text_file/<text>')
        def text2lip_file(text):
            tools.text2wav(self.tts_model, self.converter, text, tts_config, speaker_id, src_path, self.source_se,
                           self.target_se, wav_save_path)
            _, gen = tools.wav2lip_pre(wav_save_path, wav2lip_config, mel_step_size, self.full_frames,
                                       self.face_det_results)
            out = cv2.VideoWriter('temp/result.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'), wav2lip_config['fps'], (self.frame_w, self.frame_h))
            for (img_batch, mel_batch, frames, coords) in gen:
                pred = batch_eval(img_batch, mel_batch, self.wav2lip_model)

                for p, f, c in zip(pred, frames, coords):
                    y1, y2, x1, x2 = c
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    f[y1:y2, x1:x2] = p
                    out.write(f)
            out.release()
            out_dir = f"{wav2lip_config['output_dir']}/output_{speaker_key}.mp4"
            command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(wav_save_path, 'temp/result.avi', out_dir)
            subprocess.call(command, shell=platform.system() != 'Windows')
            with open(out_dir, 'rb') as f:
                video_data = base64.b64encode(f.read()).decode()

            data = {
                'video': 'data:video/mp4;base64,%s' % video_data,
            }
            json_data = json.dumps(data)
            return json_data

        @app.route('/text_stream/<text>')
        def text2lip_stream(text):
            tools.text2wav(self.tts_model, self.converter, text, tts_config, speaker_id, src_path, self.source_se,
                           self.target_se, wav_save_path)
            wav, gen = tools.wav2lip_pre(wav_save_path, wav2lip_config, mel_step_size, self.full_frames,
                                         self.face_det_results)
            full_preds = []
            for (img_batch, mel_batch, frames, coords) in gen:
                pred = batch_eval(img_batch, mel_batch, self.wav2lip_model)
                for p, f, c in zip(pred, frames, coords):
                    y1, y2, x1, x2 = c
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    f[y1:y2, x1:x2] = p
                    full_preds.append(f)
            clip_v = ImageSequenceClip(full_preds, fps=25)
            clip_a = AudioArrayClip(wav, fps=44100)
            clip_v.set_audio(clip_a)
            clip_v.write_videofile()

        self.server = WSGIServer(('', self.config['web']['port']), app)
        self.server.serve_forever()
