from fastapi import FastAPI, Request # type: ignore
from . import config # 正确示例
from fastapi.staticfiles import StaticFiles  # type: ignore
from fastapi.templating import Jinja2Templates  # type: ignore

app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend/"), name="static")

templates = Jinja2Templates(directory="frontend/static")

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)