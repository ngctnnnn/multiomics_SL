#!/bin/sh
pip3 install fair-esm
pip3 install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
pip3 install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

pip3 install openmm==7.5.1 pdbfixer cudatoolkit==11.3.* einops fairscale omegaconf hydra-core pandas pytest 
pip3 install hmmer==3.3.2 hhsuite==3.3.0 kalign2==2.04 pytorch==1.12.* 
pip3 install biopython==1.79 deepspeed==0.5.9 dm-tree==0.1.6 ml-collections==0.1.0 numpy==1.21.2 PyYAML==5.4.1 
pip3 install requests==2.26.0 scipy==1.7.1 tqdm==4.62.2 typing-extensions==3.10.0.2 pytorch_lightning==1.5.10 wandb==0.12.21