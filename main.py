import uvicorn
from typing import Optional
from fastapi import FastAPI

from experiment.frames import calculate_folder_frame_count

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/experiment/folder_frame_count/{folder_path}")
def cal_folder_frame_count(folder_path: str):
    frames_count = calculate_folder_frame_count(folder_path)
    return {"frames": frames_count}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)