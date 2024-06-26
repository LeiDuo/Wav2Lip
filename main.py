import os

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

import face_detection
import wav2lip_configuration
from models import Wav2Lip

# 环境变量需要在引入huggingface相关库之前设置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from web import Web
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS

device = "cuda:0" if torch.cuda.is_available() else "cpu"
img_size = 96


def load_from_yaml() -> dict:
    with open('model.yaml', mode='r') as file:
        config = yaml.safe_load(file)
    for key in wav2lip_configuration.default_config.keys():
        if key not in config['wav2lip']:
            config['wav2lip'][key] = wav2lip_configuration.default_config[key]
    config['wav2lip']['img_size'] = img_size
    return config


def load_tts_model(config: dict) -> (ToneColorConverter, TTS, torch.Tensor, torch.Tensor):
    tts_config = config['tts']
    os.makedirs(tts_config['output_dir'], exist_ok=True)
    tone_color_converter = ToneColorConverter(
        f"{tts_config['ckpt_converter']}/config.json", device=device
    )
    tone_color_converter.watermark_model = None
    tone_color_converter.load_ckpt(f"{tts_config['ckpt_converter']}/checkpoint.pth")
    reference_speaker = (
        tts_config['reference_speaker']  # This is the voice you want to clone
    )
    target_se, _ = se_extractor.get_se(
        reference_speaker, tone_color_converter, vad=False
    )
    # 音频输出路径
    src_path = f"{tts_config['output_dir']}/tmp.wav"
    tts_model = TTS(language="ZH", device=device, ckpt_path=f"{tts_config['ckpt_tts']}/checkpoint.pth",
                    config_path=f"{tts_config['ckpt_tts']}/config.json")
    speaker_ids = tts_model.hps.data.spk2id

    speaker_key = list(speaker_ids.keys())[0]
    speaker_id = speaker_ids[speaker_key]
    speaker_key = speaker_key.lower().replace("_", "-")
    source_se = torch.load(
        f"{tts_config['ckpt_speakers']}/{speaker_key}.pth", map_location=device
    )
    save_path = f"{tts_config['output_dir']}/output_v2_{speaker_key}.wav"
    tts_model.tts_to_file("第一次推理时普遍速度较慢，启动时先推理一遍减少延迟", speaker_id, src_path,
                          speed=tts_config['speed'])
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path
    )
    return tone_color_converter, tts_model, source_se, target_se


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images, wav2lip_config):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)

    batch_size = wav2lip_config['face_det_batch_size']

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = wav2lip_config['pads']
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not wav2lip_config['nosmooth']:
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results


def load_wav2lip_model(config: dict) -> (Wav2Lip, list, list):
    wav2lip_config = config['wav2lip']
    if os.path.isfile(wav2lip_config['face']) and wav2lip_config['face'].split('.')[1] in ['jpg', 'png', 'jpeg']:
        wav2lip_config['static'] = True

    if not os.path.isfile(wav2lip_config['face']):
        raise ValueError('face argument must be a valid path to video/image file')

    elif wav2lip_config['face'].split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(wav2lip_config['face'])]

    else:
        video_stream = cv2.VideoCapture(wav2lip_config['face'])
        wav2lip_config['fps'] = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if wav2lip_config['resize_factor'] > 1:
                frame = cv2.resize(frame, (
                    frame.shape[1] // wav2lip_config['resize_factor'],
                    frame.shape[0] // wav2lip_config['resize_factor']))

            if wav2lip_config['rotate']:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = wav2lip_config['crop']
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    print("Number of frames available for inference: " + str(len(full_frames)))

    full_frames_copy = full_frames.copy()
    if wav2lip_config['box'][0] == -1:
        if not wav2lip_config['static']:
            face_det_results = face_detect(full_frames_copy, wav2lip_config)  # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([full_frames_copy[0]], wav2lip_config)
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = wav2lip_config['box']
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames_copy]

    wav2lip_model = load_model(wav2lip_config['ckpt'])
    print("Model loaded")
    return wav2lip_model, full_frames, face_det_results


if __name__ == '__main__':
    config_m = load_from_yaml()
    tone_color_converter_m, tts_model_m, source_se_m, target_se_m = load_tts_model(config_m)
    wav2lip_model_m, full_frames_m, face_det_results_m = load_wav2lip_model(config_m)
    web = Web(tone_color_converter_m, tts_model_m, source_se_m, target_se_m, wav2lip_model_m, full_frames_m,
              face_det_results_m, config_m)
    web.start()
