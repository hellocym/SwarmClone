import os
from fastapi import FastAPI, Request  # type: ignore
from . import config  # 正确示例
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore

app = FastAPI()

# 获取当前项目的根目录
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 设置静态文件目录和模板目录
static_dir = os.path.join(base_dir, "frontend/static")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=static_dir)

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)