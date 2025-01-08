#!/bin/bash

# Copyright 2022, MetaQuotes Ltd.

# MetaTrader download url
URL="https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"
# Wine version to install: stable or devel
WINE_VERSION="stable"

#Prepare: Remove any old sources of wine
sudo rm /etc/apt/sources.list.d/winehq*

# Prepare: Install basic calculator
sudo apt update
sudo apt upgrade
sudo apt install bc

# Prepare: switch to 32 bit and add Wine key
sudo dpkg --add-architecture i386
sudo mkdir -pm755 /etc/apt/keyrings
sudo wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key

# Get Ubuntu version and trim
OS_VER=$(lsb_release -r | cut -f2)

# Multiply the version by 100 to avoid float operations
OS_VER_100=$(echo "$OS_VER * 100" | bc -l | cut -d "." -f1)

# Choose repository based on Ubuntu version
if (( $OS_VER_100 >= 2300 )); then
  sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/lunar/winehq-lunar.sources
elif (( $OS_VER_100 < 2300 )) && (( $OS_VER_100 >= 2210 )); then
  sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/kinetic/winehq-kinetic.sources
elif (( $OS_VER_100 < 2210 )) && (( $OS_VER_100 >= 2100 )); then
  sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/jammy/winehq-jammy.sources
elif (( $OS_VER_100 < 2100 )) && (($OS_VER_100 >= 2000 )); then
  sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/focal/winehq-focal.sources
elif (( $OS_VER_100 < 2000 )); then
  sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/bionic/winehq-bionic.sources
fi

# Update package and install Wine
sudo apt update
sudo apt upgrade
sudo apt install --install-recommends winehq-$WINE_VERSION

# Download MetaTrader
wget $URL

# Set environment to Windows 10
WINEPREFIX=~/.mt5 winecfg -v=win10
# Start MetaTrader installer
WINEPREFIX=~/.mt5 wine mt5setup.exe
