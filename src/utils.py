# File: utils.py
# FUNGSI: Administrasi database (setup, view, reset).

import sqlite3
from pathlib import Path
import os
import sys
import logging
from datetime import date

# --- PENTING: Import Modul Database ---
try:
    from . import _data_persistance as db
except ImportError:
    # Fallback jika dijalankan langsung
    import _data_persistance as db


# Konfigurasi Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- KONFIGURASI DATABASE ---
DATABASE_NAME = "absensi.db"
# Path yang benar: database/absensi.db
DB_PATH = Path("database") / DATABASE_NAME 

# =======================================================
# FUNGSI 1: INITIAL DB SETUP (Diambil dari initial_db_setup.py)
# =======================================================

def initialize_database():
    """
    Fungsi utama untuk membuat folder database dan tabel 'magang' dan 'absensi'.
    """
    # Pastikan folder 'database' ada
    if not DB_PATH.parent.exists():
        os.makedirs(DB_PATH.parent)
        logging.info(f"Folder database/ dibuat.")

    logging.info(f"Memulai inisialisasi database: {DB_PATH}")
    
    # Memanggil fungsi pembuatan tabel dari modul persistence
    db.create_initial_tables(str(DB_PATH))
    
    logging.info("Inisialisasi database Selesai. Tabel siap.")

# =======================================================
# FUNGSI 2: VIEW LOG (Diambil dari 04_report_viewer.py)
# =======================================================

def view_absensi_log():
    """Membaca dan mencetak semua log absensi dari tabel 'absensi'."""
    if not DB_PATH.exists():
        print(f"\n🚨 Error Database: File '{DB_PATH}' tidak ditemukan.")
        print("Pastikan Anda sudah menjalankan inisialisasi.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # Mengambil semua data dari tabel 'absensi', diurutkan berdasarkan waktu terbaru
        c.execute("SELECT id, magang_id, nama, timestamp, type FROM absensi ORDER BY timestamp DESC")
        logs = c.fetchall()
        conn.close()
        
        if not logs:
            print("\n🚨 Log Absensi masih kosong.")
            return

        print("\n" + "="*70)
        print(" " * 20 + "LAPORAN ABSENSI PT TELKOMSAT")
        print("="*70)
        print(f"| {'Log ID':<6} | {'Pegawai ID':<10} | {'Nama':<15} | {'Waktu Absensi':<20} | {'Tipe':<5} |")
        print("-" * 70)
        
        for log in logs:
            id_log, magang_id, nama, waktu, tipe = log
            print(f"| {id_log:<6} | {magang_id:<10} | {nama:<15} | {waktu:<20} | {tipe:<5} |")
        
        print("="*70 + "\n")
        
    except sqlite3.OperationalError as e:
        print(f"❌ Error Database: {e}. Pastikan nama tabel dan kolom sudah benar.")

# =======================================================
# FUNGSI 3: RESET LOG (Diambil dari 07_reset_log.py)
# =======================================================

def reset_absensi_log():
    """
    Menghapus SEMUA log absensi dan mereset penghitung ID Log.
    """
    conn = None 
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # 1. Menghapus semua baris data dari tabel utama
        c.execute("DELETE FROM absensi")
        
        # 2. Mereset penghitung auto-increment ID Log
        try:
            # DELETE FROM sqlite_sequence WHERE name='absensi'
            c.execute("DELETE FROM sqlite_sequence WHERE name='absensi'")
            print("Pesan: Penghitung ID Log (sqlite_sequence) berhasil direset.")
        except sqlite3.OperationalError:
            # Ini adalah penanganan error 'no such table' yang kita bahas sebelumnya
            print("Pesan: Tabel sqlite_sequence tidak ada (wajar jika log kosong). Melewati reset ID.")
        
        conn.commit()
        
        print("\n=============================================")
        print(f"✅ Log Absensi di '{DB_PATH}' BERHASIL di-RESET TOTAL.")
        print("=============================================")
        
    except sqlite3.OperationalError as e:
        print(f"❌ Error Database: {e}. Pastikan file database/absensi.db ada.")
    finally:
        if conn:
            conn.close()


# =======================================================
# MAIN EXECUTION
# =======================================================

if __name__ == "__main__":
    
    print("\n--- Pilihan Administrasi Database ---")
    print("1: Inisialisasi Database (Buat Tabel)")
    print("2: Tampilkan Log Absensi")
    print("3: Hapus SEMUA Log Absensi")
    
    choice = input("Masukkan pilihan (1/2/3): ")
    
    if choice == '1':
        initialize_database()
    elif choice == '2':
        view_absensi_log()
    elif choice == '3':
        confirm = input("ANDA YAKIN INGIN MENGHAPUS SEMUA LOG ABSENSI? (y/n): ")
        if confirm.lower() == 'y':
            reset_absensi_log()
        else:
            print("Penghapusan dibatalkan.")
    else:
        print("Pilihan tidak valid.")
