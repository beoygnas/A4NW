import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from pyparsing import empty
import requests
import time

def streamlit_main() : 
    with st.sidebar:
        selected = option_menu("AI for No War", ["프로젝트 데모", "프로젝트 소개", "팀원 소개"],
            icons=['postcard', 'projector', 'people'],
            menu_icon="exclamation-triangle", default_index=1,
        )
        
    if selected == "프로젝트 데모" :    
        st.markdown('# AI for No War')

        st.markdown('#### prompt_1')
        prompt_1 = st.text_input('원하는 스타일 텍스트를 입력', 'a photo of war criminal', label_visibility='collapsed', key='prompt_1')
        
        st.markdown('#### prompt_2')
        prompt_2 = st.text_input('원하는 스타일 텍스트를 입력', 'a photo of terrified person', label_visibility='collapsed', key='prompt_2')
        
        st.markdown('#### negative prompt')
        negative_prompt = st.text_input('원하는 스타일 텍스트를 입력', 'blurry, unrealistic, low res, not human, occidental, cartoon', label_visibility='collapsed', key='negative_prompt')
                
        st.markdown('#### Image')
        uploaded_image = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"], label_visibility='collapsed')
        
        if uploaded_image is not None:
            Image.open(uploaded_image).save('../fastapi/data/image.png')
            col1,col2 = st.columns([0.5,0.5])
            with col1 :
                st.image(uploaded_image, use_column_width=True)
            with col2 :
                steps_preprocessing = st.slider(
                    "preprocessing steps",
                    0, #시작 값 
                    1000, #끝 값  
                    100, # 기본값
                    step=10
                )
                steps_pnp = st.slider(
                    "pnp steps",
                    0, #시작 값 
                    100, #끝 값  
                    50, # 기본값
                    step=10
                )
                attention_threshold = st.slider(
                    "text threshold (text)",
                    0.0, #시작 값 
                    1.0, #끝 값  
                    0.8, # 기본값
                    step=0.1
                )
                feature_threshold = st.slider(
                    "feature threshold (image)",
                    0.0, #시작 값 
                    1.0, #끝 값  
                    0.8, # 기본값,
                    step=0.1
                )
            
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
            
    elif selected == "프로젝트 소개" :
        st.title("프로젝트 소개")
        st.write("~이런 프로젝트입니다.")
        
    else : 
        st.title("팀원 소개")
        st.write("~~ 입니다.")
        

if __name__ == '__main__' : 
    streamlit_main()