# AI for No War

평범한 사람의 이미지를 전쟁을 겪은 사람들의 이미지로 변환시켜주는 style trasnfer.

## PNP_diffusers

[PNP diffuser](https://github.com/MichalGeyer/pnp-diffusers) 코드를 일부 수정하여 사용하였습니다.

### Setup

```bash
conda create -n pnp-diffusers python=3.9
conda activate pnp-diffusers
pip install -r requirements.txt
```

### inference

1. `pnp_diffusers/data` 에 이미지 저장
2. `pnp_diffusers/config_pnp.yaml` 에서 config 수정
   - image : image 경로
   - prompt, negative_prompt : 스타일과 관련한 prompt
   - attention_threshold, feature_threshold : injection 관련 hyperparameter
   - steps_pnp, steps_preprocess : inversion / sampling에서 steps

```bash
cd pnp_diffusers

python3 pnp.py --config_path='config_pnp.yaml'
```

## Demo

- FE : Streamlit
- BE : fastAPI
