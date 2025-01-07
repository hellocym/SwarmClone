import asyncio
import os
from fastapi import FastAPI, Request  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore
from . import config  # 正确示例

app = FastAPI()

# 设置静态文件目录和模板目录
static_dir = "frontend/static"

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=static_dir)

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/action1")
async def action1():
    await asyncio.sleep(1)
    return {"message": "Action 1 executed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.PANEL_HOST, port=config.PANEL_WEB_PORT)