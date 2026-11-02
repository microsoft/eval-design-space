import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import REPO_ROOT, repo_path

import json
import os
from together import Together


METADATA_PATH = Path(repo_path("output_data/wang/together/batch_inputs/batch_info.json"))
client = Together(api_key=os.environ["TOGETHER_API_KEY"])
data = json.loads(METADATA_PATH.read_text())

# go in reverse order
data.reverse()
for item in data:

    batch_id = item["batch_id"]
    

    # grab the name and job value from parsing the filepath (e.g., 'output_data/armstrong/together/batch_inputs/batch_input_armstrong_armstrong_meta_llama_Meta_Llama_3.1_8B_Instruct_Turbo.jsonl')

    filename = Path(repo_path(item["file_path"])).name
    name = filename.split("_")[2]
    job = filename.split("_")[3]
    model = item["model"].split("/")[-1]
    # check if there contains a part
    if "part" in item["file_path"]:
        batch_num = Path(filename).stem.split("_")[-1]
    else: 
        batch_num = "part01"
    # file_path: "output_data/wang/together/batch_inputs/batch_input_armstrong_rozado_meta_llama_3.3_70b_instruct_turbo_part02.jsonl",

    # Note: Mistral-7B models not supported in Together AI Batch API
    if model not in ["Llama-3.3-70B-Instruct-Turbo", "Meta-Llama-3.1-8B-Instruct-Turbo", "Mistral-Small-24B-Instruct-2501"]:
        continue

    batch = client.batches.get_batch(batch_id) # :contentReference[oaicite:4]{index=4}

    # If you're on SDK v2 instead, use:

    # batch = client.batches.retrieve(batch_id) # :contentReference[oaicite:5]{index=5}

    item["status"] = batch.status

    try:
        if batch.status == "FAILED":
            # Check for error file
            # breakpoint()
            error_file_id = getattr(batch, 'error_file_id', None)
        
            if error_file_id:
                print(f"Found Error File ID: {error_file_id}")
                print("Downloading Error Log...")
            
                # Download and print the error details
                content = client.files.content(error_file_id).text
            
                print("\n--- ERROR REPORT ---")
                for line in content.split('\n'):
                    if line.strip():
                        err = json.loads(line)
                        # Use .get() to avoid crashing if fields are missing
                        print(f"Request ID: {err.get('custom_id', 'Unknown')}")
                        print(f"Message:    {err.get('error', {}).get('message')}")
                        print("-" * 30)
            else:
                print("X Batch failed, but no specific error file was generated.")
                print("This usually means the file itself was invalid (e.g., wrong file extension, empty lines).")
    
            
        else:
            print("Batch is not in FAILED state.")
    except: 
        print('error')
        continue

    # Optional: keep these if/when present

    if getattr(batch, "output_file_id", None):

        item["output_file_id"] = batch.output_file_id

    if getattr(batch, "error_file_id", None):

        item["error_file_id"] = batch.error_file_id



    print(batch.status)

    # # create the directory if it doesn't exist

    try:

        os.makedirs(repo_path(f"output_data/wang/together/batch_outputs/{model}/"), exist_ok=True)
        print(f"output_data/wang/together/batch_outputs/{model}/name_{name}_job_{job}_{batch_num}.jsonl")
        client.files.retrieve_content(id=batch.output_file_id, output=repo_path(f"output_data/wang/together/batch_outputs/{model}/name_{name}_job_{job}_{batch_num}.jsonl"))

    except Exception as e:

        print(f"Error retrieving content: {e}")

    # update the line in data with the new status
    data[data.index(item)]["status"] = batch.status





'''

mistralai/Mistral-7B-Instruct-v0.1

mistralai/Mistral-7B-Instruct-v0.1 


'''