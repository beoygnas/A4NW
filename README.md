# 🌟AI for No War

📢 **2023년 겨울학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다**

## 소개

대부분의 사람들이 전쟁을 자신과 동떨어진 사건으로 바라보고, 심각성을 잊어버리곤 합니다.

![image](https://github.com/AIKU-Official/aiku-23-2-ai-for-no-war/assets/81809224/b4e70aab-b85a-4e9c-a6fb-951c3e91fa2e)
유니세프에서 진행했던 Deep Empathy project는 이 문제를 해결하기 위해 AI 기술을 사용하였습니다. 평범한 대도시 이미지에 전쟁 중인 도시의 이미지를 합성하여 전쟁의 파괴성을 느끼도록 한것입니다.

이에 착안을 얻어, style transfer 기능을 가진 모델을 통해 평범한 사람들이 전쟁으로 인해 변화된 모습을 생성하는 프로젝트를 기획하였습니다. 이를 통해 전쟁으로 인해 사람이 어떻게 변화하는지 보여주어 전쟁의 심각성을 고발하고, 전쟁 문제를 상기시키고자 합니다.

## 방법론

- 평범한 얼굴에 '전쟁'이라는 style을 입히는 모델을 만들고자 하였습니다.
  <p align="center"><img alt="pipeline" src="assets/img1.png" width="80%" /></p>
- 위 이미지에 표현되어 있듯이, 2가지 파이프라인으로 진행하였습니다.
  1. style feature를 이미지에서 추출
  2. style feature를 텍스트에서 추출
- 구글링을 통해 전쟁을 겪은 사람들의 이미지를 수집하여 데이터셋을 구성하였습니다. dlib 라이브러리를 사용하여 얼굴 위주로만 크롭한 후 사용하였습니다.

- **Model 1. StyleGAN2-based** StyleGAN2는 Styel transfer의 대표적인 모델인 StyleGAN에서 일부 문제를 개선한 모델입니다. input image(평범한 얼굴)과 style image(전쟁 이미지)를 입력하면 style image에서 style을 추출하여 원하는 style을 가진 이미지를 생성합니다. discriminator의 판별 성능과 함께 generator가 전쟁 style에 가까운 얼굴 이미지를 생성할 수 있도록 학습합니다.
<p align="center"><img alt="Attention R2U-Net Learning Curve" src="assets/img2.png" width="80%" /></p>

- **Model 2. Diffusion-based** Plug and Play Diffusion은 input image(평범한 얼굴)과 style text(전쟁 관련 키워드)를 입력하면 text에서 style을 추출하여 원하는 style을 가진 이미지를 생성합니다. Stable-Diffusion 기반의 모델이며, 학습이 필요하지 않습니다.
<p align="center"><img alt="Robust U-Net Learning Curve" src="assets/img3.png" width="80%" /></p>

## 환경 설정

### Requirements

- requirements.txt 참고

### 데모서버 컴퓨팅환경

- OS : Ubuntu 20.04 (WSL2)
- GPU : NVIDIA geForce RTX 3060
- RAM : 32GB
- cuda 11.7

## 사용 방법

### 1. StyleGAN2-based

```bash
cd stylegan2-based
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
python stylegan_inference.py --style victim --batch 4
```

### 2. PNP_diffusers

[PNP diffuser](https://github.com/MichalGeyer/pnp-diffusers) 코드를 일부 수정하여 사용하였습니다.

**Setup**

```bash
conda create -n pnp-diffusers python=3.9
conda activate pnp-diffusers
pip install -r requirements.txt
```

**inference**

1. `pnp_diffusers/data` 에 이미지 저장
2. `pnp_diffusers/config_pnp.yaml` 에서 config 수정
   - `image` : image 경로
   - `prompt`, `negative_prompt` : 스타일과 관련한 prompt
   - `attention_threshold`, `feature_threshold` : injection 관련 hyperparameter
   - `steps_pnp`, `steps_preprocess` : inversion / sampling에서 steps

```bash
cd pnp_diffusers

python3 pnp.py --config_path='config_pnp.yaml'
```

## 프로젝트 데모

[**데모 사이트**](http://124.197.159.108:8503/)

`streamlit`, `fastapi` 라이브러리를 이용해 간단한 데모서버를 제작했습니다. batch size를 2로 하여, 한 장의 이미지에 대해 두 가지 버젼의 전쟁을 겪은 사진을 제공합니다. 넉넉치 않은 환경에서 inference 서버를 운영하고 있기 때문에, 이미지 두 장을 생성하는데에 약 1분정도의 시간이 걸립니다.

## 예시 결과

<img src="assets/img6.png" width="47%" height="45%"><img src="assets/img7.png" width="45%" height="45%">
<img src="assets/img5.png">
