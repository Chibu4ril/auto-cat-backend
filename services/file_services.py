from core.config import supabase

async def fetch_uploaded_files():
    response = supabase.storage.from_("uploads").list()

    if response is None or isinstance(response, dict) and "error" in response:
        print("Error Occurred Here! Innocent", response)
        return []

    return [
        {
            "name": file["name"],
            "url": supabase.storage.from_("uploads").get_public_url(file["name"])
        }
        for file in response
        if file["name"] != ".emptyFolderPlaceholder"
    ]


async def fetch_training_sets():
    response = supabase.storage.from_("training_set").list()

    if response is None or isinstance(response, dict) and "error" in response:
        print("Error Occurred Here! Innocent", response)
        return []

    return [
        {
            "name": file["name"],
            "url": supabase.storage.from_("training_set").get_public_url(file["name"])
        }
        for file in response
        if file["name"] != ".emptyFolderPlaceholder"
    ]



   
async def delete_uploaded_files(fileUrl):
    from urllib.parse import urlparse
    parsed_url = urlparse(fileUrl)
    file_path = parsed_url.path.lstrip("/")  

    response = await supabase.storage.from_("uploads").remove([file_path])

    print("Supabase delete response:", response)

    # Log the full response for debugging
    print("Supabase delete response:", response)
    
    # Check if there was an error in the response
    if response.get("error"):
        return {"success": False, "message": f"Error deleting file: {response['error']}"}
    
    return {"success": True, "message": f"File '{file_path}' deleted successfully"}
    

