import sys
import subprocess
import os

requirements = {
    "pip": {
        "general": [
            "http://pixelstonelab.com:5005/sc_cosyvoice-0.1.0-py3-none-any.whl",
            "spacy-pkuseg",
            "dragonmapper",
            "hanziconv",
            "spacy",
            "textgrid",
            "pygame",
            "zhconv"
        ],
        "linux": [
            "http://pixelstonelab.com:5005/ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl",
            "http://pixelstonelab.com:5005/ttsfrd_dependency-0.1-py3-none-any.whl",
        ],
        "windows": []
    },
    "conda": {
        "general": [
            "montreal-forced-aligner" 
        ],
        "linux": [],
        "windows": [
            "pynini==2.1.6"
        ]
    }
}

print(""" * 开始为 SwarmClone 添加以下依赖: \n\t 1. TTS CosyVoice \n\t 2. Montreal Forced Aligner""")
install = input(" * 确认? [y/n]")

def install_conda_packages(packages, channel):
    # 获取 conda 的路径
    condapath = os.popen("where conda").read().strip()
    if not condapath:
        print("无法找到 conda.exe，请检查 Conda 的安装路径.")
        return
    
    # 调用 conda install
    install_cmd = [condapath, "install", "-c", channel] + packages + ["-y"]
    subprocess.run(install_cmd)

def install_pip_packages(packages):
    # 调用 pip install
    install_cmd = ["python", "-m", "pip", "install"] + packages
    subprocess.run(install_cmd)

if install.strip().lower() == "y":
    print(" * 开始添加 Minimum CosyVoice: ")
    os_system = sys.platform
    if os_system.startswith("linux"):
        print(" * 安装 Linux 平台依赖中: ")
        if len(requirements["pip"]["linux"]) > 0:
            install_pip_packages(requirements["pip"]["linux"])
        if len(requirements["conda"]["linux"]) > 0:
            install_conda_packages(requirements["conda"]["linux"], "conda-forge")
    else:
        print(" * 安装 Windows 平台依赖中: ")
        if len(requirements["pip"]["windows"]) > 0:
            install_pip_packages(requirements["pip"]["windows"])
        if len(requirements["conda"]["windows"]) > 0:
            install_conda_packages(requirements["conda"]["windows"], "conda-forge")

    print(" * 安装通用依赖中: ")
    install_pip_packages(requirements["pip"]["general"])
    install_conda_packages(requirements["conda"]["general"], "conda-forge")
    print(" # 添加完毕！")
else:
    print(" # 取消添加")