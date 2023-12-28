import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from pyparsing import empty
import requests
import time

def streamlit_main() : 
    with st.sidebar:
        selected = option_menu("AI for No War", ["프로젝트 데모", "프로젝트 소개", "모델 구조", "팀원 소개"],
            icons=['postcard', 'projector', 'people'],
            menu_icon="exclamation-triangle", default_index=1,
        )
        
    if selected == "프로젝트 데모" :    
        st.markdown('# AI for No War')

        # st.markdown('#### prompt_1')
        # prompt_1 = st.text_input('원하는 스타일 텍스트를 입력', 'a photo of traumatic, miserable, suffered, painful, wounded person', label_visibility='collapsed', key='prompt_1')
        # st.markdown('#### prompt_2')
        # prompt_2 = st.text_input('원하는 스타일 텍스트를 입력', 'a caricature of traumatic, miserable, suffered, painful, wounded person', label_visibility='collapsed', key='prompt_2')
        # st.markdown('#### negative prompt')
        # negative_prompt = st.text_input('원하는 스타일 텍스트를 입력', 'blurry, unrealistic, low res, not human, cartoon', label_visibility='collapsed', key='negative_prompt')

        prompt_1 = 'a photo of traumatic, miserable, suffered, painful, wounded person'
        prompt_2 = 'a caricature of traumatic, miserable, suffered, painful, wounded person'
        negative_prompt = 'blurry, unrealistic, low res, not human, cartoon'
            
        st.write('변화시킬 이미지를 선택해주세요. 얼굴이 정면을 바라보고 정방형 이미지일수록 좋습니다.')
        uploaded_image = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"], label_visibility='collapsed')
        
        if uploaded_image is not None:
            Image.open(uploaded_image).save('../fastapi/data/image.png')
            col1,col2 = st.columns([0.5,0.5])
            with col1 :
                st.image(uploaded_image, use_column_width=True)
            with col2 :
                st.write('두 장의 이미지를 생성하는데 30초 정도 소모됩니다.\n')
                st.write('1. 첫번째 사진은 전쟁으로 인한 트라우마, 절망적이고 부상으로 고통받고 있는 이미지를 생성합니다.')
                st.write('2. 두번째 사진은 같은 프롬프트의 사진이지만, 캐리커쳐 스타일의 이미지를 생성합니다.')
                st.write(' ')
                st.write(' ')
                threshold = st.slider(
                    "threshold (원본이미지 반영 hyperparameter)",
                    0.0, #시작 값 
                    1.0, #끝 값  
                    0.8, # 기본값
                    step=0.1
                )
                st.write('threshold는 1에 가까울수록 원본이미지의 형태를 잘 보존하지만 실험적으로는 0.8이 최적의 값이었습니다. 결과가 마음에 들지 않을 경우 조정해주세요.')
                # steps_preprocessing = st.slider(
                #     "preprocessing steps",
                #     0, #시작 값 
                #     100, #끝 값  
                #     50, # 기본값
                #     step=10
                # )
                # steps_pnp = st.slider(
                #     "pnp steps",
                #     0, #시작 값 
                #     50, #끝 값  
                #     25, # 기본값
                #     step=5
                # )
            
            steps_preprocessing = 50
            steps_pnp = 25
            attention_threshold = threshold
            feature_threshold = threshold
        
        if st.button('Generate') :
            data = {
                "image" : 'data/image.png',
                "prompt" : [prompt_1, prompt_2],
                "negative_prompt" : negative_prompt,
                "steps_preprocessing" : steps_preprocessing,
                "steps_pnp" : steps_pnp,
                "attention_threshold" : attention_threshold,
                "feature_threshold" : feature_threshold
            }
            print(data)
            
            response = requests.post('http://localhost:8502/inference', json=data)
            # time_passed = 0
            # with st.spinner(f'{time_passed} / 70s'):
            #     for i in range(70) : 
            #         time_paseed = i
            #         time.sleep(1)
                    
            img0 = response.json()['generated_img_0']
            img1 = response.json()['generated_img_1']
            generated_image_0 = Image.open(f'../fastapi/{img0}')
            generated_image_1 = Image.open(f'../fastapi/{img1}')
            
            # progress_text = "Operation in progress. Please wait."
            # my_bar = st.progress(0, text=progress_text)
            col1,col2 = st.columns([0.5,0.5])
            with col1 :
                st.image(generated_image_0, use_column_width=True)
            with col2 : 
                st.image(generated_image_1, use_column_width=True)
                
            st.markdown("<h3 style='text-align: center; color: red;'>인류가 전쟁을 끝내지 않으면, 전쟁이 인류를 끝낼 것이다.</h3>", unsafe_allow_html=True)
            
    elif selected == "프로젝트 소개" :
        st.title("프로젝트 소개")
        st.markdown("📢 **2023년 겨울학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다** ([프로젝트 GitHub](https://github.com/AIKU-Official/aiku-23-2-ai-for-no-war))")
        st.image('./assets/img7.png')
        st.write(" ")
        st.write(" ")
        st.write("대부분의 사람들이 전쟁을 자신과 동떨어진 사건으로 바라보고, 심각성을 깨닫지 못하고 있습니다. 유니세프에서 진행했던 Deep Empathy project는 이 문제를 해결하기 위해 AI 기술을 사용하였습니다. 평범한 대도시 이미지에 전쟁 중인 도시의 이미지를 합성하여 전쟁의 파괴성을 느끼도록 한것입니다.")
        st.image('./assets/img5.png')
        st.write("이에 착안을 얻어, 본 프로젝트는 style transfer 기능을 가진 모델을 통해 전쟁을 겪은 평범한 사람들의 모습을 생성합니다. 이를 통해 사람들이 전쟁이 사람을 얼마나 괴롭게 만드는지 깨닫게 하여 전쟁 문제를 상기시키고, 심각성을 느끼도록 합니다.")
        st.image('./assets/img1.png')
    
    elif selected == "모델 구조": 
        st.title("모델 구조")
        st.write("본 프로젝트에서는 다음과 같이 GAN, diffusion base의 모델을 각각 구현했습니다.")
        st.markdown("### StyleGAN2 based")
        st.image('./assets/img2.png')
        st.write(" ")
        st.write(" ")
        st.markdown("### PNP-Diffusers based")
        st.image('./assets/img3.png')
        st.write("본 데모에서는 pnp-diffusers 라는 diffusion 라이브러리를 사용하여 전쟁 style tranfser를 구현했습니다.")
        
        
    else : 
        st.title("팀원 소개")
        st.write("프로젝트에 참여한 팀원입니다.")
        col1,col2,col3,col4 = st.columns([0.25,0.25,0.25,0.25])
        with col1 :
            st.image("./assets/goo_b.png", use_column_width=True)
            st.write("구은아")
        with col2 : 
            st.image("./assets/sy_b.png", use_column_width=True)
            st.write("김상엽")
        with col3 :
            st.image("./assets/lee_b.png", use_column_width=True)
            st.write("이진규")
        with col4 :
            st.image("./assets/jung_b.png", use_column_width=True)
            st.write("정성연")
            
        st.markdown("<h3 style='text-align: center; color: red;'>\"모든 인류 죄악의 총합은 전쟁이다.\"</h3>", unsafe_allow_html=True)
        st.write(' ')
        st.write(' ')
            
        col1,col2,col3,col4 = st.columns([0.25,0.25,0.25,0.25])
        with col1 :
            st.image("./assets/goo_a.png", use_column_width=True)
            st.write("통계학과 19")
        with col2 : 
            st.image("./assets/sy_a.png", use_column_width=True)
            st.write("컴퓨터학과 18")
        with col3 :
            st.image("./assets/lee_a.png", use_column_width=True)
            st.write("건축사회환경공학부 18")
        with col4 :
            st.image("./assets/jung_a.png", use_column_width=True)
            st.write("바이오시스템의과학부 19")

        

if __name__ == '__main__' : 
    streamlit_main()