echo '== Hardware / OS =='
cat /proc/device-tree/model
uname -m
cat /etc/nv_tegra_release
dpkg-query -W -f='${Package} ${Version}\n' nvidia-l4t-core 2>/dev/null
cat /etc/os-release | grep PRETTY_NAME

echo '== Resources =='
free -h
df -h /
sudo nvpmodel -q 2>/dev/null || true

echo '== Docker GPU runtime =='
docker --version
docker info --format '{{.Runtimes}}'
nvidia-ctk --version 2>/dev/null || true

echo '== TensorRT =='
trtexec --version 2>/dev/null || /usr/src/tensorrt/bin/trtexec --version 2>/dev/null || true
python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null || true

echo '== Optional RealSense =='
lsusb | grep -i realsense || true
ls -l /dev/video* 2>/dev/null || true
