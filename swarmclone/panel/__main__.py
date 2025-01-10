import asyncio
import os
from fastapi import FastAPI, Request  # type: ignore
from fastapi.responses import RedirectResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore
from . import config  # 正确示例

app = FastAPI()

# 设置静态文件目录和模板目录
static_dir = "frontend/static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=static_dir)

init_success = False

@app.get("/")
async def root(request: Request):
    # 如果初始化成功，直接跳转到面板
    if init_success:
        return RedirectResponse(url="/panel")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/panel_init")
@app.get("/panel_init")
async def panel_init(request: Request):
    global init_success
    if init_success:
        return RedirectResponse(url="/panel")
    # TODO: 在这里放置一些面板初始化代码
    await asyncio.sleep(3)
    init_success = True
    return {"status": "success"}

@app.get("/panel")
async def panel_main(request: Request):
    return templates.TemplateResponse("panel.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.PANEL_HOST, port=config.PANEL_WEB_PORT)