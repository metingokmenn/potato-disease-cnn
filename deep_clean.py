import tensorflow as tf
import os

DATASET_DIR = "dataset"

def validate_images():
    print(f"--- DERİNLEMESİNE TEMİZLİK BAŞLIYOR: {DATASET_DIR} ---")
    print("TensorFlow ile her dosya tek tek okunup test ediliyor...")
    
    deleted_count = 0
    checked_count = 0
    
    # Tüm klasörleri gez
    for root, dirs, files in os.walk(DATASET_DIR):
        for filename in files:
            file_path = os.path.join(root, filename)
            checked_count += 1
            
            # 1. Aşama: Sistem dosyalarını affetme
            if filename in ["Thumbs.db", ".DS_Store"] or filename.startswith("._"):
                try:
                    os.remove(file_path)
                    print(f"[SİLİNDİ - SİSTEM DOSYASI]: {filename}")
                    deleted_count += 1
                except: pass
                continue

            # 2. Aşama: Boş dosyaları (0 byte) sil
            if os.path.getsize(file_path) == 0:
                try:
                    os.remove(file_path)
                    print(f"[SİLİNDİ - BOŞ DOSYA]: {filename}")
                    deleted_count += 1
                except: pass
                continue

            # 3. Aşama: TensorFlow ile okumayı dene (Asıl Test)
            try:
                file_contents = tf.io.read_file(file_path)
                # Sadece decode etmeye çalış, çizmeye gerek yok
                _ = tf.io.decode_image(file_contents, channels=3, expand_animations=False)
            except Exception as e:
                # Eğer TensorFlow hata verirse, o dosya çöptür. Sil.
                print(f"[SİLİNDİ - BOZUK İÇERİK]: {filename}")
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except: pass

            # İlerleme çubuğu niyetine
            if checked_count % 1000 == 0:
                print(f"Kontrol edilen: {checked_count} dosya...")

    print(f"\n--- TARAMA BİTTİ ---")
    print(f"Toplam Silinen Dosya: {deleted_count}")
    print("Artık main.py %100 çalışacaktır.")

if __name__ == "__main__":
    validate_images()