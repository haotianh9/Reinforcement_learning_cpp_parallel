#### Some difference with totally c++ version
1. Version of libtorch, which should the same as the python one.
wget https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.10.0%2Bcpu.zip
pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
2. Replace all `#include <torch/torch.h>` with `#include <torch/extension.h>`
3. Some pytorch version need to add `TORCH_USE_RTLD_GLOBAL=YES` as an environment variable for python file execution.