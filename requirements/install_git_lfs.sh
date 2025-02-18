#!/bin/bash

cd $HOME
wget https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz
tar -xvf git-lfs-linux-amd64-v3.5.1.tar.gz
cd git-lfs-3.5.1/

chmod +x install.sh
sed -i 's|^prefix="/usr/local"$|prefix="$HOME/.local"|' install.sh

mkdir -p ~/.local/bin/
export PATH="$HOME/.local/bin:$PATH"
./install.sh
git-lfs --version