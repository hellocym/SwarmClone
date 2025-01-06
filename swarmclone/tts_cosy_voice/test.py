import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'third_party/Matcha-TTS'))

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio


base_dir = os.path.dirname(__file__)
pretrained_model_path = os.path.join(base_dir, 'pretrained_models', 'CosyVoice-300M-SFT')
prompt_wav_path = os.path.join(base_dir, 'zero_shot_prompt.wav')


cosyvoice = CosyVoice(pretrained_model_path)
# zero_shot usage, <|zh|><|en|><|jp|><|yue|><|ko|> for Chinese/English/Japanese/Cantonese/Korean
prompt_speech_16k = load_wav(prompt_wav_path, 16000)
# for i, j in enumerate(cosyvoice.inference_zero_shot('我是莫莫娅莫娅！客人晚上好呀！', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False)):
#     torchaudio.save('zero_shot_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)

outputs = list(cosyvoice.inference_zero_shot('我是莫莫娅莫娅！客人晚上好呀！', '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False))
print(len(outputs), type(outputs[0]))
for k, v in outputs[0].items():
    print(type(k), type(v), k, v)

