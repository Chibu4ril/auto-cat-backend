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
import uuid





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

def analyze_catalog_dataframe(df: pd.DataFrame) -> dict:

    # 1. Count of duplicate rows
    duplicate_rows = df.duplicated().sum()

    # 2. Total missing (NaN) cells
    total_missing_cells = df.isna().sum().sum()

    # 3. Empty strings or NaNs in RRP
    rrp_missing = df["RRP"].isna().sum()
    if df["RRP"].dtype == object:
        rrp_missing += (df["RRP"].astype(str).str.strip() == "").sum()

    # 4. RRP = 0
    rrp_zero = 0
    try:
        rrp_zero = (pd.to_numeric(df["RRP"], errors="coerce") == 0).sum()
    except Exception:
        pass

    # 5. NaN count per column
    nan_per_column = df.isna().sum().to_dict()


    return {
        "duplicate_rows": int(duplicate_rows),
        "total_missing_cells": int(total_missing_cells),
        "rrp_empty_or_missing": int(rrp_missing),
        "rrp_zero": int(rrp_zero),
        "nan_per_column": {col: int(count) for col, count in nan_per_column.items()},
        "total_rows": len(df),
        "total_columns": len(df.columns)
    }


def upload_dataframe_to_supabase(df):
    records = df.to_dict(orient='records')
    print(records[:5])  # Print first 5 records for debugging
    chunk_size = 500
    for i in range(0, len(records), chunk_size):
        chunk = records[i:i + chunk_size]
        response = supabase.table("cleaned_products").insert(chunk).execute()
        if response.status_code != 201:
            print(f"Error inserting chunk {i//chunk_size + 1}: {response.data}")
        else:
            print(f"Chunk {i//chunk_size + 1} inserted successfully")


def generate_fake_ean(prod_no: str, rrp: float) -> str:
    """Simulate EAN using hash of SKU and RRP"""
    base = f"{prod_no}{rrp}"
    return str(abs(hash(base)))[:13] 

def clean_catalog_dataframe(df):
    try:
        # Step 1: Rename columns to normalized names
        column_mapping = {
            "ProdNr": "prodNo",
            "ManProdNr": "manufProdNo",
            "Manuf_Name": "manufName",
            "Product_Desc": "prodDesc",
            "TradePrice": "tradePrice",
            "RRP": "rrp",
            "currency_code": "currencyCode",
            "file_creation": "filecreation",
            "Stock": "stock",
            "Stock_Delivery_Date": "stockdelvdate",
            "Classification": "classification",
            "E_Orderable": "eorderable",
            "ManufProdURL": "manufProdUrl",
            "ProductFamilies": "prodfamilies",
            "AdvancedClassification": "advClassification",
            "FutExp_1": "futexp1",
            "FutExp_2": "futexp2",
            "FutExp_3": "futexp3",
            "FutExp_4": "futexp4",
            "FutExp_5": "futexp5",
            "Weight": "weight",
        }
        df = df.rename(columns=column_mapping)

        # Step 2: Remove duplicates
        df = df.drop_duplicates()
        
        # Step 3: Trim whitespace from all string columns
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)


         # Step 5: Ensure correct data types
        string_cols = [ "prodNo", "manufProdNo", "manufName", "prodDesc", "currencyCode", "filecreation", "stockdelvdate", "classification", "eorderable", "manufProdUrl", "prodfamilies", "advClassification", "futexp1", "futexp2",
                       "futexp3", "futexp4", "futexp5"]
        
        numeric_cols = [ "tradePrice", "rrp", "stock", "weight"]
        
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
                
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Step 6: Fill NaN values
        df[string_cols] = df[string_cols].fillna("N/A")
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Step 7: Set selling price
        if "rrp" in df.columns and "tradePrice" in df.columns:
            df["price"] = df["rrp"]
            df.loc[df["rrp"] == 0, "price"] = df["tradePrice"] * 1.20

        # Step 8: Generate unique `id`
        df["id"] = [str(uuid.uuid4()) for _ in range(len(df))]


        # Step 7: Ensure EAN exists and fill in blanks
        if "EAN" not in df.columns:
            df["EAN"] = "N/A"
        df["EAN"] = df["EAN"].astype(str)

        df["EAN"] = df.apply(
            lambda row: generate_fake_ean(row["prodNo"], row["rrp"]) if row["EAN"] in ["", "N/A", "nan"] else row["EAN"], axis=1)
        
        # print(df.columns)
        df.columns = df.columns.str.lower()

        # print(df.head())

        upload_dataframe_to_supabase(df)

        return df
    except Exception as e:
        print(f"Error cleaning catalog dataframe: {str(e)}")
        return {"error": str(e)}


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
        
        report = analyze_catalog_dataframe(df)
        # print(report)

        clean_catalog_dataframe(df)

        return {"data": report}
    
    except Exception as e:
        return {"error": str(e)}
     

@router.get("/health")
def health_check():
    return {"status": "ok"}