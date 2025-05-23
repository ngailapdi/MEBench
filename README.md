# MEBench
### Data generation
Please see [data_generation/README.md]() for more details.
### Evaluating VLMs
#### Step 1: Install environment
```
conda env create -f environment.yml
```
#### Step 2: Inference
LISA
```
python scripts/me_test.py --model lisa --image_size 224 --data_mode DATA_MODE --model_max_length 512 --from_pretrained xinlai/LISA-13B-llama2-v1 --vision-tower openai/clip-vit-large-patch14
```
CogVLM
```
python scripts/me_test.py --model cogvlm --from_pretrained ~/.cache/huggingface/hub/models--THUDM--cogvlm-grounding-generalist-hf/snapshots/f3440bb2344cf1fe226857042c5a1ea323a9a0fb/ --image_size 1024 --data_mode DATA_MODE
```
Sa2VA
```
python scripts/me_test.py --model sa2va --from_pretrained ByteDance/Sa2VA-4B --torch-dtype bf16 --image_size 224 --data_mode DATA_MODE
```
OMG-LLaVA
```
python scripts/me_test.py  --config ~/develop/OMG-Seg/omg_llava/omg_llava/configs/finetune/omg_llava_7b_finetune_8gpus.py 	--pth_model ~/develop/OMG-Seg/omg_llava/pretrained/omg_llava/omg_llava_7b_finetune_8gpus.pth --model omg_llava --image_size=256 --data_mode DATA_MODE
```
F-LMM
```
python scripts/me_test.py --config ~/develop/F-LMM/configs/deepseek_vl/frozen_deepseek_vl_7b_chat_unet_sam_l_refcoco_png.py --pth_model ~/develop/F-LMM/checkpoints/frozen_deepseek_vl_7b_chat_unet_sam_l_refcoco_png.pth --image_size 224 --data_mode DATA_MODE --model flmm
```
Gemini
```
python scripts/me_test.py --model gemini --image_size 1000 --data_mode DATA_MODE
```

