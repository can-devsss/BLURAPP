import os
import uvicorn

if __name__ == "__main__":
    # Render requires binding to 0.0.0.0 and using its injected PORT.
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
