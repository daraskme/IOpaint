import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_plugins_package():
    install("onnxruntime>=1.20")
    if sys.version_info >= (3, 11):
        install("rembg[cpu]>=2.0.78")
    else:
        # rembg >=2.0.70 requires Python >=3.11.
        install("rembg[cpu]==2.0.69")
