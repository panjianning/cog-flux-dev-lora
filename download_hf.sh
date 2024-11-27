sudo apt-get install git-lfs -y
sudo git lfs install
sudo git lfs clone https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter
sudo git lfs clone https://huggingface.co/google/siglip-so400m-patch14-384

sudo curl -o /usr/local/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
sudo chmod +x /usr/local/bin/cog
