#!/usr/bin/env python3
"""
Simple Test Version for Azure Web App
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from datetime import datetime
import os

app = FastAPI(title="HackRx Test", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "HackRx API is running", "endpoint": "/hackrx/run"}

@app.post("/hackrx/run")
async def hackrx_run(request: Request):
    """Test endpoint"""
    try:
        data = await request.json()
        return {
            "message": "Test endpoint working", 
            "timestamp": datetime.now().isoformat(),
            "received_data": data
        }
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 