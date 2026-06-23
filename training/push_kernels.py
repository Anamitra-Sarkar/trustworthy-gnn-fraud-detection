import os
import re
import subprocess
import shutil

def main():
    token_path = "/home/anamitra/Downloads/API_Keys_and_Secrets/hf_token"
    if not os.path.exists(token_path):
        print("HF token not found at", token_path)
        return
        
    with open(token_path, "r") as f:
        token = f.read().strip()
        
    # Scripts to push
    scripts = [
        ("kaggle_train_gnn.py", "gnn-kernel-metadata.json"),
        ("kaggle_train_edl.py", "edl-kernel-metadata.json")
    ]
    
    # We must push them sequentially
    for script_name, metadata_name in scripts:
        print(f"Preparing to push {script_name} using metadata {metadata_name}...")
        
        # 1. Back up original script
        backup_name = script_name + ".bak"
        shutil.copy(script_name, backup_name)
        
        # 2. Back up original kernel-metadata.json
        meta_backup = None
        if os.path.exists("kernel-metadata.json"):
            meta_backup = "kernel-metadata.json.bak"
            shutil.copy("kernel-metadata.json", meta_backup)
            
        try:
            # 3. Read and replace placeholder in script
            with open(script_name, "r") as f:
                content = f.read()
                
            modified_content = content.replace("PLACEHOLDER_HF_TOKEN", token)
            
            with open(script_name, "w") as f:
                f.write(modified_content)
                
            # 4. Copy metadata to kernel-metadata.json
            shutil.copy(metadata_name, "kernel-metadata.json")
            
            # 5. Push to Kaggle
            print(f"Pushing {script_name} to Kaggle...")
            res = subprocess.run(["kaggle", "kernels", "push", "-p", "."], capture_output=True, text=True)
            print("Stdout:", res.stdout)
            print("Stderr:", res.stderr)
            
        finally:
            # 6. Restore original script
            if os.path.exists(backup_name):
                shutil.move(backup_name, script_name)
                
            # 7. Restore original kernel-metadata.json
            if meta_backup and os.path.exists(meta_backup):
                shutil.move(meta_backup, "kernel-metadata.json")
            elif os.path.exists("kernel-metadata.json"):
                os.remove("kernel-metadata.json")
                
        print(f"Finished pushing {script_name}.\n")

if __name__ == "__main__":
    main()
