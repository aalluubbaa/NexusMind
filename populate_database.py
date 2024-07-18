import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from PyPDF2 import PdfReader, PdfWriter

CHROMA_PATH = "chroma"
DATA_PATH = "data"
SPLIT_PDF_PATH = "split_pdfs"

def split_pdf(input_path, output_folder, pages_per_section=250):
    reader = PdfReader(input_path)
    total_pages = len(reader.pages)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    split_files = []
    for start in range(0, total_pages, pages_per_section):
        end = min(start + pages_per_section, total_pages)
        output = PdfWriter()
        
        for page in range(start, end):
            output.add_page(reader.pages[page])
        
        output_filename = f"{os.path.splitext(os.path.basename(input_path))[0]}_part_{start//pages_per_section + 1}.pdf"
        output_path = os.path.join(output_folder, output_filename)
        
        with open(output_path, "wb") as output_file:
            output.write(output_file)
        
        split_files.append(output_filename)
        print(f"Created: {output_filename}")

    print(f"Split complete. {len(split_files)} files created in {output_folder}")
    return split_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    if args.reset:
        print("✨ Clearing Database")
        clear_database()
    
    unembedded_files = get_unembedded_files()
    if not unembedded_files:
        print("All files have been embedded.")
        restart = input("Do you want to restart the embedding process? (yes/no): ").lower()
        if restart == 'yes':
            clear_database()
            unembedded_files = get_unembedded_files()
        else:
            print("Exiting the script.")
            return

    split_files = split_large_pdfs(unembedded_files)
    process_split_files(split_files)

def get_unembedded_files():
    all_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
    return [f for f in all_files if not f.startswith("(embedded)")]

def split_large_pdfs(files_to_process, max_pages=500):
    if not os.path.exists(SPLIT_PDF_PATH):
        os.makedirs(SPLIT_PDF_PATH)
    
    all_split_files = []
    for filename in files_to_process:
        file_path = os.path.join(DATA_PATH, filename)
        reader = PdfReader(file_path)
        if len(reader.pages) > max_pages:
            print(f"Splitting large PDF: {filename}")
            split_files = split_pdf(file_path, SPLIT_PDF_PATH, max_pages)
            all_split_files.extend(split_files)
        else:
            all_split_files.append(filename)
    
    return all_split_files

def process_split_files(split_files):
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    
    for i, file in enumerate(split_files, 1):
        print(f"{i}. {file}")
    
    while split_files:
        user_input = input("Enter the numbers of the PDFs to process (e.g., '1,3-5,7' or 'all' for all remaining): ")
        if user_input.lower() == 'all':
            selected_files = split_files.copy()
        else:
            selected_files = parse_user_input(user_input, split_files)
        
        for file in selected_files:
            process_single_pdf(file, db)
            mark_as_embedded(file)
            split_files.remove(file)
        
        if not split_files:
            print("All files have been processed.")
            break

def parse_user_input(user_input, split_files):
    selected_indices = set()
    for part in user_input.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            selected_indices.update(range(start-1, end))
        else:
            selected_indices.add(int(part) - 1)
    
    return [split_files[i] for i in selected_indices if i < len(split_files)]

def process_single_pdf(file, db):
    print(f"Processing {file}...")
    file_path = os.path.join(DATA_PATH, file) if file in os.listdir(DATA_PATH) else os.path.join(SPLIT_PDF_PATH, file)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    
    if new_chunks:
        print(f"Adding {len(new_chunks)} new chunks from {file}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print(f"No new chunks to add from {file}")

def mark_as_embedded(file):
    if file in os.listdir(DATA_PATH):
        old_path = os.path.join(DATA_PATH, file)
        new_name = f"(embedded){file}"
        new_path = os.path.join(DATA_PATH, new_name)
        os.rename(old_path, new_path)
    elif file in os.listdir(SPLIT_PDF_PATH):
        os.remove(os.path.join(SPLIT_PDF_PATH, file))
    print(f"Marked {file} as embedded.")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # Reset all file names to remove (embedded) prefix
    for filename in os.listdir(DATA_PATH):
        if filename.startswith("(embedded)"):
            old_path = os.path.join(DATA_PATH, filename)
            new_path = os.path.join(DATA_PATH, filename.replace("(embedded)", ""))
            os.rename(old_path, new_path)
    
    # Remove split PDFs
    if os.path.exists(SPLIT_PDF_PATH):
        shutil.rmtree(SPLIT_PDF_PATH)

if __name__ == "__main__":
    main()