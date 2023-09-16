import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from demo import getData
import uuid
import io

app = FastAPI()

origins = ["*"]

VIDEODIR = "videos/"

@app.get("/check_connection")
def check_connection():
    connected = True 
    return {"connected": connected}

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):

    file.filename = f"{uuid.uuid4()}.mp4"
    contents = await file.read()

    #save the file
    with open(f"{VIDEODIR}{file.filename}", "wb") as f:
        f.write(contents)

    response = getData(file.filename)
    print(response)
    with open('demo/' + file.filename, mode="rb") as video_file:
        video_data = video_file.read()

    # return response
    return StreamingResponse(io.BytesIO(video_data), media_type="video/mp4")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)