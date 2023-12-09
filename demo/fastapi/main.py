import os, sys
from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import JSONResponse
from pnp_diffusers.pnp_utils import seed_everything
from pnp_diffusers.pnp import PNP, get_timesteps
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

app = FastAPI()

class Item(BaseModel) : 
    image : str
    prompt : list
    negative_prompt : str
    steps_preprocessing: int
    steps_pnp: int
    attention_threshold: float
    feature_threshold: float
    
# ---pnp---
model_config = {
    'seed': 1, 
    'device': 'cuda', 
    'sd_version': '2.1',
    'guidance_scale': 7.5
}
pnp = PNP(model_config)

@app.get("/")
async def root():
	return "메인화면 streamlit"

@app.post("/test")
async def root(item: Item):
    dicted_item = dict(item)
    dicted_item['success'] = True
    print(dicted_item)
    response_dict = {}
    response_dict['status'] = 200
    response_dict['generated_img'] = 'data/image.png'
    

    return JSONResponse(response_dict)
	# return {"status" : 200, "generated_img" : generated_image}

@app.post("/inference")
async def inference(item: Item):
    
    dicted_item = dict(item)
    img_path = dicted_item['image']
    seed_everything(model_config["seed"])
    pnp.img_path = img_path
    # preprocess config -> streamlit에서 입력으로 받을 수 있으면 좋겠음.
    pnp.extract_latents(num_steps=dicted_item['steps_preprocessing'],
                        inversion_prompt="",
                        extract_reverse=False)
    pnp_config = {
        'prompt': dicted_item['prompt'],  
        'negative_prompt': dicted_item['negative_prompt'], 
        'n_timesteps': dicted_item['steps_pnp'],
        'pnp_attn_t': dicted_item['attention_threshold'], 
        'pnp_f_t': dicted_item['feature_threshold']
    }
    generated_img0, generated_img1 = pnp.run_pnp(pnp_config)
    response_dict = {}
    response_dict['status'] = 200
    response_dict['generated_img_0'] = generated_img0
    response_dict['generated_img_1'] = generated_img1

    return JSONResponse(response_dict)