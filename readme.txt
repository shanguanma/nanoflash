a nano flash attention with pure cutlass cute dsl, Hope it's clear and simple, easy to learn and use

# Installation

Option 1: Using uv
git clone https://github.com/shanguanma/nanoflash.git
cd nanoflash

uv venv --python 3.12
source .venv/bin/activate
uv pip install nvidia-cutlass-dsl
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118



how to debug
export CUTE_DSL_LOG_TO_CONSOLE=10

it wiil generate log txt, its name is CUTE_DSL.log  

Arch of Both RTX3090  and A100 is ampere 
