# File: 01_initial_db_setup.py
# FUNGSI: HANYA untuk membuat tabel 'magang' dan 'absensi' di database 'absensi.db'.
# Dijalankan HANYA SEKALI di Server.

import os
import sys
import logging

# Menambahkan path folder saat ini agar modul _data_persistance.py ditemukan
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modul persistence data
import _data_persistance as db_mod

# Konfigurasi Logging
logging.basicConfig(level=logging.INFO)

# --- KONFIGURASI ---
DATABASE_NAME = "database/absensi.db"

def initialize_database():
    """
    Fungsi utama untuk membuat tabel 'magang' dan 'absensi'.
    """
    logging.info(f"Memulai inisialisasi database: {DATABASE_NAME}")
    
    # Memanggil fungsi pembuatan tabel dari modul persistence
    db_mod.create_initial_tables(DATABASE_NAME)
    
    logging.info("Inisialisasi database Selesai. Tabel siap.")

if __name__ == "__main__":
    initialize_database()