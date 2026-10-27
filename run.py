import uvicorn

if __name__ == "__main__":
    # Run from the folder containing main.py, processor.py and index.html.
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
