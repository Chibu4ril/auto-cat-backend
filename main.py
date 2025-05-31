from fastapi import FastAPI
from api.routes import router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="FastAPI CSV Processor")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000/*", "http://localhost:8000/*"],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


app.include_router(router, prefix="/api")

@app.get("/")
def home():
    return {"message": "Welcome to the FastAPI File Service"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    # To start the server, navigate to the directory containing main.py and run the following command:
    # Ensure you are in the directory containing main.py and run:
    # uvicorn main:app --host 0.0.0.0 --port 8000 --reload