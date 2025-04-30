import os

def clear_cached_encodings(dataset_dirs):
    """
    dataset_dirs: list of dataset directories to search (e.g., ["dataset", "dataset_inst", "dataset_vocal"])
    """
    deleted = 0
    for dataset_dir in dataset_dirs:
        if not os.path.exists(dataset_dir):
            continue
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file == "cached_encoding.pt":
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        deleted += 1
                        print(f"🗑️ Deleted {file_path}")
                    except Exception as e:
                        print(f"⚠️ Failed to delete {file_path}: {e}")

    print(f"\n✅ Done. Deleted {deleted} cached encoding files.")

if __name__ == "__main__":
    # define which datasets you want to clear
    dataset_dirs = ["dataset", "dataset_inst", "dataset_vocal"]
    
    clear_cached_encodings(dataset_dirs)
