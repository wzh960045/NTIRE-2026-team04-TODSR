# [NTIRE 2026 Challenge on Mobile Real-World Image Super-Resolution](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)
# Team 04 Model TODSR
[![ntire](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fraw.githubusercontent.com%2Fzhengchen1999%2FNTIRE2025_ImageSR_x4%2Fmain%2Ffigs%2Fdiamond_badge.json)](https://www.cvlai.net/ntire/2026/)
[![page](https://img.shields.io/badge/Project-Page-blue?logo=github&logoSvg)](https://gobunu.github.io/ntire_mobile_sr)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=jiatongli2024.NTIRE2026_Mobile_RealWorld_ImageSR&right_color=violet)](https://github.com/jiatongli2024/NTIRE2026_Mobile_RealWorld_ImageSR)
[![GitHub Stars](https://img.shields.io/github/stars/jiatongli2024/NTIRE2026_Mobile_RealWorld_ImageSR?style=social)](https://github.com/jiatongli2024/NTIRE2026_Mobile_RealWorld_ImageSR)


### Environments
### Command to calculate metrics
```bash
conda create -n nitre python=3.10
conda activate nitre
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

Select the model you would like to test:

```bash
CUDA_VISIBLE_DEVICES=5 python test.py --test_dir /data0/wangzihao/nitremobsr/test --save_dir result/test --model_id 4

Command to calculate metrics

```bash
CUDA_VISIBLE_DEVICES=5 python eval.py \
--output_folder "/data0/wangzihao/nitrewzh/result/test/0004_TODSR/test" \
--target_folder "/data0/wangzihao/nitremobsr/HR" \
--metrics_save_path "./IQA_results" \
--gpu_ids 0 \
```

The `eval.py` file accepts the following 4 parameters:

- `output_folder`: Path where the restored images are saved.
- `target_folder`: Path to the HR images in the `test` dataset. This is used to calculate FR-IQA metrics.
- `metrics_save_path`: Directory where the evaluation metrics will be saved.
- `device`: Computation devices. For multi-GPU setups, use the format `0,1,2,3`.

## Reference Code
We provide a [reference implementation for checkpoint saving](./uitls/ref_ckpt_save.py), which we will use to reproduce participants’ experimental results.
Participants may use our implementation as-is or modify it based on our reference code.

## License and Acknowledgement
This code repository is release under [MIT License](LICENSE). 
