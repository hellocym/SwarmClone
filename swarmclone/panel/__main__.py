from fastapi import FastAPI, Request
from swarmclone.utils import config
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()
app.mount("/static", StaticFiles(directory="frontend/"), name="static")

templates = Jinja2Templates(directory="frontend/static")

@app.get("/")
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)