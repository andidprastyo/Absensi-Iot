import sqlite3
import numpy as np
import logging
from datetime import datetime, date # Import datetime dan date untuk penanganan waktu
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)

# --- FUNGSI UTAMA DATABASE ---

def create_initial_tables(db_name: str):
    """
    Membuat tabel 'magang' dan 'absensi' jika belum ada.
    Dipanggil oleh initial_db_setup.py.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # 1. Tabel MAGANG (Master Data Wajah)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS magang (
                id TEXT PRIMARY KEY,
                nama TEXT NOT NULL,
                master_embedding BLOB NOT NULL,
                last_updated TEXT
            );
        """)

        # 2. Tabel ABSENSI (Log Absensi Harian)
        # Catatan: AUTOINCREMENT dihilangkan karena INTEGER PRIMARY KEY sudah menyediakan fungsi yang sama
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS absensi (
                id INTEGER PRIMARY KEY,
                magang_id TEXT NOT NULL,
                nama TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                type TEXT DEFAULT 'MASUK'
            );
        """)
        conn.commit()
        logging.info(f"Tabel 'magang' dan 'absensi' berhasil dibuat atau sudah ada di {db_name}.")

    except sqlite3.Error as e:
        logging.error(f"Error saat membuat tabel: {e}")
    finally:
        if conn:
            conn.close()

def clear_magang_table(db_name: str):
    """Hapus semua data di tabel magang."""
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM magang")
        # Mereset counter ID (hanya jika ada)
        try:
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='magang'")
        except sqlite3.OperationalError:
            pass # Lewati jika tabel belum ada
        conn.commit()
        logging.info("Tabel magang berhasil dibersihkan dan counter ID direset.")
    except sqlite3.Error as e:
        logging.error(f"Gagal membersihkan tabel magang: {e}")
    finally:
        if conn:
            conn.close()

def save_new_magang(db_name: str, magang_id: str, magang_name: str, embedding_array: np.ndarray) -> tuple:
    """
    Menyimpan atau memperbarui data magang (termasuk master embedding) ke database.
    Dipanggil oleh training_embedding.py.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Konversi numpy array (float32) ke byte (BLOB)
        embedding_blob = embedding_array.astype(np.float32).tobytes()
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Menggunakan INSERT OR REPLACE untuk memastikan ID yang sama akan diupdate
        cursor.execute("""
            INSERT OR REPLACE INTO magang (id, nama, master_embedding, last_updated)
            VALUES (?, ?, ?, ?)
        """, (magang_id, magang_name, embedding_blob, current_time))

        conn.commit()
        return True, f"Data Magang '{magang_name}' berhasil disimpan/diperbarui."

    except sqlite3.Error as e:
        return False, f"Gagal menyimpan data magang: {e}"
    finally:
        if conn:
            conn.close()

def load_all_magang_embeddings(db_name: str) -> List[Dict[str, Any]]:
    """
    Memuat semua master embedding dari database untuk digunakan oleh Server (FaceNet Recognition).
    Dipanggil saat startup server_main.py.
    """
    conn = None
    magang_list = []
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("SELECT id, nama, master_embedding FROM magang")
        
        for row in cursor.fetchall():
            magang_id, nama, embedding_blob = row
            
            # Konversi BLOB kembali ke numpy array (float32, 512 dimensi)
            embedding_array = np.frombuffer(embedding_blob, dtype=np.float32)
            
            magang_list.append({
                'id': magang_id,
                'name': nama,
                'embedding': embedding_array
            })
            
    except sqlite3.Error as e:
        logging.error(f"Gagal memuat embedding: {e}")
    finally:
        if conn:
            conn.close()
    
    return magang_list

def check_already_absen(db_name: str, magang_id: str, log_type: str = 'MASUK') -> bool:
    """
    Memeriksa apakah magang sudah absen (MASUK) hari ini.
    """
    conn = None
    try:
        # Gunakan hari ini (tanpa jam) sebagai batas bawah
        today_start = date.today().strftime('%Y-%m-%d 00:00:00')
        
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Cari log absensi hari ini dengan tipe MASUK
        cursor.execute("""
            SELECT 1 FROM absensi 
            WHERE magang_id = ? AND type = ? AND timestamp >= ? 
            LIMIT 1
        """, (magang_id, log_type, today_start))
        
        return cursor.fetchone() is not None

    except sqlite3.Error as e:
        logging.error(f"Gagal memeriksa status absensi: {e}")
        return False

    finally:
        if conn:
            conn.close()

def update_absensi_log(db_name: str, magang_id: str, magang_name: str, log_time: str, log_type: str = 'MASUK'):
    """
    Mencatat absensi berhasil ke tabel absensi menggunakan waktu eksplisit dari server.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO absensi (magang_id, nama, timestamp, type)
            VALUES (?, ?, ?, ?)
        """, (magang_id, magang_name, log_time, log_type))

        conn.commit()
        logging.info(f"Log absensi '{log_type}' untuk {magang_name} berhasil dicatat pada {log_time}.")

    except sqlite3.Error as e:
        logging.error(f"Gagal mencatat log absensi: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    logging.warning("Modul persistence ini biasanya di-import, bukan dijalankan langsung.")