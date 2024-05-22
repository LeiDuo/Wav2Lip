import cv2
import numpy as np
from melo.api import TTS
from openvoice.api import ToneColorConverter

import audio


def text2wav(tts_model: TTS, converter: ToneColorConverter, text: str, tts_config: dict, speaker_id, src_path,
             source_se, target_se, wav_save_path):
    tts_model.tts_to_file(text, speaker_id, src_path,
                          speed=tts_config['speed'])
    converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=wav_save_path
    )


def wav2lip_pre(wav_save_path: str, wav2lip_config: dict, mel_step_size: int, full_frames: list,
                face_det_results: list):
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
        face_det_results_seg = face_det_results[0]
    else:
        face_det_results_seg = face_det_results[:len(mel_chunks)]
    gen = datagen(full_frames, mel_chunks, face_det_results_seg, wav2lip_config)
    return wav, gen


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
