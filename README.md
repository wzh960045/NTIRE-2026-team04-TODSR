# [NTIRE 2026 Challenge on Mobile Real-World Image Super-Resolution](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)
# Team 04 Model TODSR
[![ntire](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fraw.githubusercontent.com%2Fzhengchen1999%2FNTIRE2025_ImageSR_x4%2Fmain%2Ffigs%2Fdiamond_badge.json)](https://www.cvlai.net/ntire/2026/)
[![page](https://img.shields.io/badge/Project-Page-blue?logo=github&logoSvg)](https://gobunu.github.io/ntire_mobile_sr)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=jiatongli2024.NTIRE2026_Mobile_RealWorld_ImageSR&right_color=violet)](https://github.com/jiatongli2024/NTIRE2026_Mobile_RealWorld_ImageSR)
[![GitHub Stars](https://img.shields.io/github/stars/jiatongli2024/NTIRE2026_Mobile_RealWorld_ImageSR?style=social)](https://github.com/jiatongli2024/NTIRE2026_Mobile_RealWorld_ImageSR)


### Environments

```bash
conda create -n nitre python=3.10
conda activate nitre
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### Prepare SD model

Download the pretrained SD-2.1-base models from [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).
or 
```bash
(optional)export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download stabilityai/stable-diffusion-2-1-base --local-dir stable-diffusion-2-1-base
```bash
should put the "stable-diffusion-2-1-base" into "model_zoo"

### Select the model you would like to test:

```bash
CUDA_VISIBLE_DEVICES=5 python test.py --test_dir [test] --save_dir [result] --model_id 4
```
### Command to calculate metrics
```bash
CUDA_VISIBLE_DEVICES=5 python eval.py \
--output_folder "[result]" \
--target_folder "[HR]" \
--metrics_save_path "./IQA_results" \
--gpu_ids 0 \
```
eval.py have handle Div2k filename 
result:0801x4.png
HR:0801.png

## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
