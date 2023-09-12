import uvicorn
from typing import Optional, Union
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware 
import json
import uuid

from mmaction.apis import inference_recognizer, init_recognizer
from experiment.frames import calculate_folder_frame_count

app = FastAPI()

origins = ["*"]

VIDEODIR = "videos/"

# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.post("/SlowFastResult")
# def SlowFastResult(file: UploadFile):
#     data = json.load(file.file.read())
#     return {"content":data,"filename":file.filename}

# @app.get("/experiment/folder_frame_count/{folder_path}")
# def cal_folder_frame_count(folder_path: str):
#     frames_count = calculate_folder_frame_count(folder_path)
#     return {"frames": frames_count}

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):

    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    #save the file
    with open(f"{VIDEODIR}{file.filename}", "wb") as f:
        f.write(contents)

    config_path = './configs/bmn_2xb8-400x100-9e_activitynet-feature.py'
    checkpoint_path = 'bmn_2xb8-400x100-9e_activitynet-feature_20220908-79f92857.pth' # 可以是本地路径
    img_path = '/videos/' + file.filename   # 您可以指定自己的图片路径

    # 从配置文件和权重文件中构建模型
    model = init_recognizer(config_path, checkpoint_path, device="cuda:0")  # device 可以是 'cuda:0'
    # 对单个视频进行测试
    #inference_recognizer() 跑不起來
    result = inference_recognizer(model, img_path)

    return {"filename": img_path}



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)