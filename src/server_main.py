import os
import sys
import time
import io
import json
import numpy as np
import torch
import logging
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

from fastapi import FastAPI, File, UploadFile, status, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from gtts import gTTS # Digunakan untuk simulasi audio
import httpx # Digunakan untuk API calls (jika diperlukan)

# --- PENTING: Import Modul Database ---
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import _data_persistance as db

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- KONFIGURASI SERVER & AI ---
DATABASE_NAME = 'absensi.db'
app = FastAPI(title="Absensi Magang Telkomsat AI Server")
SIMILARITY_THRESHOLD = 0.95  # Cosine Distance: semakin kecil angkanya semakin mirip (0 = sempurna)
tts_output_folder = "audio_responses"

resnet = None
mtcnn = None
MAGANG_EMBEDDINGS = []

# --- HELPER FUNCTIONS ---

def calculate_cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Menghitung Cosine Distance (1 - Cosine Similarity). Jarak terdekat = 0."""
    dot_product = np.dot(emb1.flatten(), emb2.flatten())
    norm_emb1 = np.linalg.norm(emb1)
    norm_emb2 = np.linalg.norm(emb2)
    
    if norm_emb1 == 0 or norm_emb2 == 0:
        return 1.0 

    cosine_similarity = dot_product / (norm_emb1 * norm_emb2)
    return 1 - cosine_similarity

def recognize_face(live_embedding: np.ndarray):
    """Mengidentifikasi wajah dengan Cosine Distance terendah."""
    if not MAGANG_EMBEDDINGS:
        return "UNKNOWN", 999.0, "UNKNOWN"

    min_distance = float('inf')
    best_match_id = "UNKNOWN"
    best_match_name = "UNKNOWN"

    for magang in MAGANG_EMBEDDINGS:
        master_embedding = magang['embedding'].flatten()
        distance = calculate_cosine_distance(live_embedding, master_embedding)
        
        if distance < min_distance:
            min_distance = distance
            best_match_id = magang['id']
            best_match_name = magang['name']

    if min_distance > SIMILARITY_THRESHOLD:
        return "UNKNOWN", min_distance, "UNKNOWN"
    else:
        return best_match_id, min_distance, best_match_name

def generate_audio(text_message: str, magang_id: str) -> str:
    """Membuat file audio MP3 (TTS) dari pesan."""
    try:
        audio_filename = f"response_audio_{magang_id}_{int(time.time())}.mp3"
        audio_path = os.path.join(tts_output_folder, audio_filename)
        
        os.makedirs(tts_output_folder, exist_ok=True)
        
        # Menggunakan gTTS untuk membuat audio MP3
        tts = gTTS(text=text_message, lang='id')
        tts.save(audio_path)
        
        # Mengembalikan URL relatif
        return f"/audio/{audio_filename}"
        
    except Exception as e:
        logging.error(f"Gagal membuat audio TTS: {e}")
        return ""


# --- LIFECYCLE HOOKS ---

@app.on_event("startup")
async def startup_event():
    """Memuat model AI dan data master saat Server dimulai."""
    global resnet, mtcnn, MAGANG_EMBEDDINGS
    
    logging.info("Memulai Server. Memuat Model AI (MTCNN & FaceNet) dan Data Master...")
    
    # 1. Inisialisasi Device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Menggunakan Device: {device}")

    # 2. Muat MTCNN dan FaceNet
    mtcnn = MTCNN(
        image_size=160, margin=14, min_face_size=20, 
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, device=device
    )
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # 3. Muat Data Master Embedding dari DB
    MAGANG_EMBEDDINGS = db.load_all_magang_embeddings(DATABASE_NAME)
    
    if not MAGANG_EMBEDDINGS:
        logging.warning("Database Master Magang KOSONG. Jalankan 02_training_embedding.py!")
    
    logging.info(f"Inisialisasi Selesai. Total {len(MAGANG_EMBEDDINGS)} data magang dimuat.")

@app.on_event("shutdown")
def shutdown_event():
    """Hapus folder audio saat server dimatikan."""
    import shutil
    try:
        if os.path.exists(tts_output_folder):
            shutil.rmtree(tts_output_folder)
            logging.info("Folder audio_responses berhasil dihapus.")
    except Exception as e:
        logging.error(f"Gagal menghapus folder audio: {e}")


# --- ENDPOINT FASTAPI ---

@app.post("/absensi")
async def absensi_endpoint(file: UploadFile = File(...)):
    """
    Menerima gambar full frame dari Klien ESP32, memprosesnya, dan memberikan respons audio.
    """
    start_time = time.time()
    
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes))
        
        # 2. Deteksi dan Ekstraksi Wajah (MTCNN)
        face_tensor = mtcnn(img) 
        
        if face_tensor is None:
            message = "Maaf, wajah tidak terdeteksi dengan jelas. Coba lagi."
            audio_url = generate_audio(message, "no_face")
            return JSONResponse(status_code=400, content={
                "status": "FAIL", "message": message,
                "audio_url": audio_url,
                "processing_time": f"{time.time() - start_time:.2f}s"
            })
        
        # 3. Ekstraksi Live Embedding (FaceNet)
        with torch.no_grad():
            live_embedding = resnet(face_tensor.unsqueeze(0)).cpu().detach().numpy()
            
        # 4. Pengenalan Wajah dan Keputusan Log
        id_magang, distance, nama = recognize_face(live_embedding)

        if id_magang != "UNKNOWN":
            # Wajah Dikenali
            db.update_absensi_log(DATABASE_NAME, id_magang, nama)
            
            message = f"Absensi masuk berhasil, selamat bekerja {nama}."
            audio_url = generate_audio(message, id_magang)
            
            logging.info(f"✅ ABSENSI BERHASIL: {nama} (Dist: {distance:.4f})")
            
            return JSONResponse(status_code=200, content={
                "status": "SUCCESS", "message": message,
                "magang_id": id_magang,
                "distance": f"{distance:.4f}",
                "audio_url": audio_url,
                "processing_time": f"{time.time() - start_time:.2f}s"
            })
        else:
            # Wajah TIDAK Dikenali
            message = f"Maaf, wajah tidak dikenali. Jarak terdekat: {distance:.2f}. Silakan coba lagi."
            audio_url = generate_audio(message, "unknown")
            
            logging.warning(f"❌ ABSENSI GAGAL: Tidak Dikenal (Dist: {distance:.4f})")

            return JSONResponse(status_code=403, content={
                "status": "FAIL", "message": message,
                "distance": f"{distance:.4f}",
                "audio_url": audio_url,
                "processing_time": f"{time.time() - start_time:.2f}s"
            })

    except Exception as e:
        error_msg = f"Error Server Internal: {e}"
        logging.error(error_msg, exc_info=True)
        # TTS Fallback
        audio_url = generate_audio("Terjadi kesalahan di server. Silakan coba lagi.", "server_error")
        raise HTTPException(status_code=500, detail=error_msg)

# --- ENDPOINT UNTUK AUDIO (Diakses oleh Klien ESP32) ---

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Melayani file audio MP3 yang dibuat oleh TTS ke Klien."""
    file_path = os.path.join(tts_output_folder, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File audio tidak ditemukan.")

    def file_iterator():
        with open(file_path, mode="rb") as f:
            yield from f
    
    return StreamingResponse(file_iterator(), media_type="audio/mp3")

@app.get("/")
async def root():
    return {"message": "Server Absensi AI Telkomsat Berjalan."}