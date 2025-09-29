# File: training_embedding.py
# Tugas: Melatih FaceNet untuk membuat embedding wajah dari dataset dan menyimpannya ke database.
# Struktur Dataset yang Didukung: data_master/dataset/Institusi/Nama Individu/gambar.jpg

import os
import sys
import numpy as np
import torch
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import logging

# --- PENTING: Import Modul Database (Gunakan impor relatif) ---
# Impor relatif bekerja jika file dijalankan sebagai bagian dari package (misalnya dari direktori utama: python -m src.training_embedding)
# Jika dijalankan langsung (python src/training_embedding.py), pastikan Anda berada di folder 'src'
try:
    from . import _data_persistance as db
except ImportError:
    # Fallback jika dijalankan langsung dari folder src
    import _data_persistance as db


# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- KONFIGURASI ---
DATABASE_NAME = 'absensi.db'
# Gunakan os.path.join untuk keamanan path
DATABASE_PATH = os.path.join("database", DATABASE_NAME) 
DATA_ROOT = 'data_master/dataset' # Folder utama dataset
BATCH_SIZE = 32 # Jumlah gambar yang diproses bersamaan

def get_magang_folders(root_dir):
    """
    Menemukan semua folder Magang (level 3) dalam struktur bertingkat 
    (data_master/dataset/Institusi/Nama Individu).
    """
    magang_data = [] # List of {'name': 'Said', 'path': '.../Said'}
    
    # os.walk akan menelusuri folder secara rekursif
    for root, dirs, files in os.walk(root_dir):
        # Kita hanya tertarik pada folder yang berisi file gambar (individu magang)
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            # Nama folder individu adalah nama magang
            magang_name = os.path.basename(root)
            # Menggunakan ID yang lebih spesifik jika diperlukan, saat ini ID=Nama
            magang_data.append({'id': magang_name, 'name': magang_name, 'path': root}) 

    return magang_data

def process_and_save_embeddings():
    """
    Memuat model, memproses dataset, menghitung embedding, dan menyimpan ke DB.
    """
    start_time = time.time()
    
    if not os.path.exists(DATA_ROOT):
        logging.error(f"Folder dataset tidak ditemukan di {DATA_ROOT}. Buat folder dan isi gambar.")
        return

    # Inisialisasi Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Memulai Training. Device: {device}")
    
    # Inisialisasi Model AI
    mtcnn = MTCNN(
        image_size=160, margin=14, min_face_size=20, 
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, 
        device=device
    )
    # Pindahkan model ke device yang sesuai
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device) 

    magang_list = get_magang_folders(DATA_ROOT)
    
    if not magang_list:
        logging.warning("Tidak ada folder individu magang yang ditemukan. Cek struktur folder.")
        return

    # Sebelum memulai, bersihkan dan buat ulang file DB jika belum ada
    if not os.path.exists(os.path.dirname(DATABASE_PATH)):
        os.makedirs(os.path.dirname(DATABASE_PATH))

    # Bersihkan tabel master wajah
    db.clear_magang_table(DATABASE_PATH)
    logging.info(f"Database magang lama dibersihkan.")
    
    total_wajah_diproses = 0

    for magang in magang_list:
        magang_id = magang['id']
        magang_name = magang['name']
        folder_path = magang['path']
        
        logging.info(f"\n--- Memproses Magang: {magang_name} ---")
        
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            logging.error(f"Tidak ada gambar ditemukan di {folder_path}. Skip.")
            continue
            
        embeddings = []
        for i in range(0, len(image_files), BATCH_SIZE):
            batch_files = image_files[i:i + BATCH_SIZE]
            batch_images = [Image.open(f).convert('RGB') for f in batch_files]
            
            try:
                # MTCNN mendeteksi dan melakukan preprocessing
                faces = mtcnn(batch_images) 
                # Filter None values (wajah yang gagal dideteksi)
                faces = [f for f in faces if f is not None] 

                if not faces:
                    logging.warning(f"Wajah tidak terdeteksi di batch {i//BATCH_SIZE + 1}. Skip.")
                    continue
                
                # Stack faces menjadi satu tensor dan pindahkan ke device
                face_tensor = torch.stack(faces).to(device)
                
                # InceptionResnetV1 menghitung embedding
                with torch.no_grad():
                    batch_embeddings = resnet(face_tensor).detach().cpu().numpy()
                
                embeddings.extend(batch_embeddings)
                
                logging.info(f"Batch {i//BATCH_SIZE + 1} diproses. {len(faces)} wajah di-'embed'.")

            except Exception as e:
                logging.error(f"ERROR saat memproses batch: {e}. Skip batch.")
                continue 

        if not embeddings:
            logging.error(f"Gagal membuat embedding untuk {magang_name}. Tidak ada wajah valid terdeteksi.")
            continue
            
        # Hitung rata-rata embedding dari semua gambar yang valid
        avg_embedding = np.mean(embeddings, axis=0)
        total_wajah_diproses += len(embeddings)
        
        # Simpan ke DB
        status, message = db.save_new_magang(DATABASE_PATH, magang_id, magang_name, avg_embedding)
        logging.info(f"Status DB: {status}. Master Embedding tersimpan.")

    logging.info(f"\n--- RINGKASAN TRAINING ---")
    if total_wajah_diproses > 0:
        logging.info(f"🎉 TRAINING SELESAI. Total {len(magang_list)} magang diproses.")
        logging.info(f"Total gambar wajah yang berhasil diproses: {total_wajah_diproses}")
    else:
        logging.error("TRAINING GAGAL: Tidak ada data magang yang valid untuk disimpan.")

    logging.info(f"Total Waktu Training: {time.time() - start_time:.2f} detik.")

if __name__ == "__main__":
    process_and_save_embeddings()
