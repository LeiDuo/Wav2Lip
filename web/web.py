import cv2
import numpy as np
import torch
import audio
import subprocess
import platform
from flask import Flask
from melo.api import TTS
from openvoice.api import ToneColorConverter
from gevent.pywsgi import WSGIServer

from models import Wav2Lip

device = "cuda:0" if torch.cuda.is_available() else "cpu"


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
            wav = audio.load_wav(wav_save_path, 16000)
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
            # full_frames_copy = self.full_frames.copy()
            gen = datagen(self.full_frames, mel_chunks, face_det_results_seg, wav2lip_config)
            frame_h, frame_w = self.full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'), wav2lip_config['fps'], (frame_w, frame_h))
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
                    out.write(f)

            out.release()

            command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(wav_save_path, 'temp/result.avi',
                                                                          "resource/wav2lip.mp4")
            subprocess.call(command, shell=platform.system() != 'Windows')

        self.server = WSGIServer(('', self.config['web']['port']), app)
        self.server.serve_forever()
