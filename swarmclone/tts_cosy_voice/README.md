# 补充依赖
在安装了 requirements_light.txt 后，还需要安装 `swarmclone/tts_cosy_voice/dependency`  
目录下的两个 whl 文件，执行以下命令进行安装：  

`pip install ttsfrd_dependency-0.1-py3-none-any.whl`  
`pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl`  

1. 在使用前，请 `cd ./swarmclone/tts_cosyvoice`  
2. 下载预训练模型 
`git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git pretrained_models/CosyVoice-300M-SFT`  
`git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd`  
`cd pretrained_models/CosyVoice-ttsfrd/`
`unzip resource.zip -d .`
 