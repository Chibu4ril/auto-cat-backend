import logging
import subprocess
from fastapi import APIRouter, HTTPException, Form
from pydantic import BaseModel
from services.file_services import fetch_uploaded_files
from services.file_services import fetch_training_sets
from services.file_services import delete_uploaded_files
import json
import sys
import os
from core.config import supabase
import pandas as pd
import io
import requests
from fastapi.responses import JSONResponse
import numpy as np
import orjson



router = APIRouter()

class FileDeleteRequest(BaseModel):
    fileUrl : str

@router.get("/files")
async def get_uploaded_files():
    try:
        upload_files = await fetch_uploaded_files()
        return {"files": upload_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete")
async def match_the_file(request: FileDeleteRequest):
    try:
        delete_result = delete_uploaded_files(request.fileUrl)
        if not delete_result:
            raise HTTPException(status_code=404, detail="File not found or could not be deleted")

        return {"success": True, "message": "File deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class FileRequest(BaseModel):
    file_path: str


@router.post("/parse-csv-from-supabase")
async def parse_csv_from_supabase(req: FileRequest):
    try:
        # Build the public URL
        public_url = supabase.storage.from_("newfiles").get_public_url(req.file_path)
        if not public_url:
            return {"error": "Could not generate public URL for the file."}
        
        print(f"Fetching file from: {public_url}")

        

        # Download the file using requests
        response = requests.get(public_url)
        if response.status_code != 200:
            return {"error": "Failed to fetch file from Supabase storage."}
        
        file_bytes = response.content


        file_name = req.file_path.lower()

        # Parse file based on extension
        if file_name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
        elif file_name.endswith(".tsv"):
            df = pd.read_csv(io.BytesIO(file_bytes), sep="\t", encoding="utf-8")
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return {"error": "Unsupported file type"}
        
        df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        )
        REQUIRED_COLUMNS = {
            "sku": "prodNo",
            "name": "manufName",  # use as a fallback
            "description": "prodDesc",
            "trade_price": "tradePrice",
            "rrp": "rrp",
            "stock": "stock",
            "ean": "EAN"
            }

        records = df.to_dict(orient="records")
        print(orjson.dumps(records[:3], option=orjson.OPT_INDENT_2).decode())

        return {"data": records}
    
    except Exception as e:
        return {"error": str(e)}
     

@router.get("/health")
def health_check():
    return {"status": "ok"}