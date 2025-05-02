# Asynchronous Pipeline Parallel

This is the official implementation of the ICML 2025 paper "Nesterov Method for Asynchronous Pipeline Parallel Optimization".

## Citation
```
@article{ajanthan2025asyncpp,
  title={Nesterov Method for Asynchronous Pipeline Parallel Optimization},
  author={Ajanthan, Thalaiyasingam and Ramasinghe, Sameera and Avraham, Gil and Zuo, Yan and Long, Alexander},
  journal={ICML},
  year={2025}
}
```

## How to run
First install `requirements.txt` and simply run the bash script `run.bash`.
This script assumes an instance with at least 8 GPUs and runs our method with 8 pipeline stages.
Tested on PyTorch 2.5.1, CUDA 12.6, and Python 3.12.

## Credits
- [PipeDream](https://github.com/msr-fiddle/pipedream)
- [PiPPy](https://pytorch.org/docs/2.5/distributed.pipelining.html)
- [NanoGPT](https://github.com/karpathy/nanoGPT)

## License
Copyright © Pluralis Research. All rights reserved.

This project is licensed under the MIT License. See the LICENSE file for details.
