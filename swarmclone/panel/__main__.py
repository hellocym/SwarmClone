from robyn import Robyn
from . import config

app = Robyn(__file__)


@app.get("/")
async def h(request):
    return "Hello, world!"


if __name__ == "__main__":
    app.start(host=config.HOST, port=config.PORT)
