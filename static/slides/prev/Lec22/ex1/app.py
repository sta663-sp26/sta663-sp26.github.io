from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/add")
async def add(x: int, y: int = 0):
    return {"result": x+y}

@app.get("/user/{user_id}")
async def user_id(user_id: int, name: str | None = None):
    res = {"user_id": user_id}
    if name is not None:
      res["name"] = name
    
    return res
