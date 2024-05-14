import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
import time

ckpt_converter = "checkpoints_v2/converter"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = "outputs_v2"

tone_color_converter = ToneColorConverter(
    f"{ckpt_converter}/config.json", device=device
)
tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

os.makedirs(output_dir, exist_ok=True)

reference_speaker = (
    "resources/example_reference.mp3"  # This is the voice you want to clone
)
target_se, audio_name = se_extractor.get_se(
    reference_speaker, tone_color_converter, vad=False
)

from melo.api import TTS

src_path = f"{output_dir}/tmp.wav"

text = "金杯里装的名酒，每斗要价十千；玉盘中盛的精美菜肴，收费万钱。胸中郁闷啊，我停杯投箸吃不下；拔剑环顾四周，我心里委实茫然。想渡黄河，冰雪堵塞了这条大川；要登太行，莽莽的风雪早已封山。像姜尚垂钓溪，闲待东山再起；伊尹乘舟梦日，受聘在商汤身边。何等艰难！何等艰难！歧路纷杂，真正的大道究竟在哪边？相信总有一天，能乘长风破万里浪；高高挂起云帆，在沧海中勇往直前。"
# Speed is adjustable
speed = 0.8

model = TTS(language="ZH", device=device, config_path="checkpoints_v2/config.json",
            ckpt_path="checkpoints_v2/checkpoint.pth", )
speaker_ids = model.hps.data.spk2id

speaker_key = list(speaker_ids.keys())[0]
speaker_id = speaker_ids[speaker_key]
speaker_key = speaker_key.lower().replace("_", "-")

source_se = torch.load(
    f"checkpoints_v2/base_speakers/ses/{speaker_key}.pth", map_location=device
)
save_path = f"{output_dir}/output_v2_{speaker_key}.wav"

for _ in range(10):
    tts_begin = time.time()
    model.tts_to_file(text, speaker_id, src_path, speed=speed)
    print(f"tts cost {time.time() - tts_begin}")

    # Run the tone color converter
    encode_message = "@MyShell"
    convert_begin = time.time()
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message,
    )
    print(f"convert cost {time.time() - convert_begin}")
