#!/bin/bash
# install_ros_gazebo_wsl.sh
# ==============================================================================
# Script to automate the installation of ROS2 Humble and Gazebo Classic 11
# on Ubuntu 22.04 LTS (WSL2 / Linux).
# ==============================================================================

set -e

echo "[STEP 1] Checking Ubuntu Version..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "$VERSION_CODENAME" != "jammy" ]; then
        echo "[WARNING] This script is designed for Ubuntu 22.04 (Jammy). Proceeding anyway..."
    fi
else
    echo "[ERROR] Cannot determine OS version. Please run on Ubuntu Linux."
    exit 1
fi

echo "[STEP 2] Setting up UTF-8 Locale..."
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

echo "[STEP 3] Adding ROS2 APT Repository..."
sudo apt install -y software-properties-common
sudo add-apt-repository -y universe
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo mkdir -p /root/.gnupg && sudo chmod 700 /root/.gnupg
# Download the verified ROS2 GPG key directly from the keyserver web endpoint using curl
sudo curl -k -sSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF42ED6FBAB17C654" | gpg --dearmor | sudo tee /usr/share/keyrings/ros-archive-keyring.gpg > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

echo "[STEP 4] Updating package index..."
sudo apt update

echo "[STEP 5] Installing ROS2 Humble & Gazebo 11..."
sudo apt install -y \
  ros-humble-desktop \
  ros-humble-gazebo-ros-pkgs \
  gazebo \
  python3-colcon-common-extensions \
  python3-pip

# Add sourcing to .bashrc if not already present
if ! grep -q "source /opt/ros/humble/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
fi

echo "[STEP 6] Installing Python Dependencies..."
pip3 install numpy pillow stable-baselines3 gym matplotlib plotly

echo "=============================================================================="
echo " ROS2 Humble and Gazebo 11 installation completed successfully!"
echo " Please run: source ~/.bashrc"
echo " To build your project, run: colcon build --packages-select drone_navigation"
echo "=============================================================================="
