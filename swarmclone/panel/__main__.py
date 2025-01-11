import asyncio
import os

from fastapi import FastAPI, Request  # type: ignore
from fastapi.responses import RedirectResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates

from . import config  # 正确示例
from swarmclone.panel.gen_api_key import *  # type: ignore


app = FastAPI()
init_success = False
api_key = None

# 设置静态文件目录和模板目录
static_dir = "frontend/static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=static_dir)


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/panel_init")
async def panel_init(request: Request):
    global init_success
    if init_success:
        return RedirectResponse(url="/panel")
    
    # TODO: 在这里放置一些面板初始化代码

    init_success,api_key,db_connection = initialize_api_key();
    
    return {"status": "success", "api_key": api_key}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.PANEL_HOST, port=config.PANEL_WEB_PORT)