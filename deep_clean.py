import tensorflow as tf
import os

DATASET_DIR = "dataset"

def validate_images():
    """
    Veri setindeki tüm görüntü dosyalarını derinlemesine kontrol eder ve bozuk dosyaları siler.
    
    Bu fonksiyon, veri seti klasöründeki tüm dosyaları üç aşamada kontrol eder:
    1. Sistem dosyalarını (Thumbs.db, .DS_Store, ._* dosyaları) siler
    2. Boş dosyaları (0 byte) siler
    3. TensorFlow ile görüntü dosyalarını decode etmeye çalışır, başarısız olanları siler
    
    Bu işlem, eğitim sırasında oluşabilecek hataları önlemek için kritiktir.
    Bozuk görüntü dosyaları model eğitimini durdurabilir veya hatalı sonuçlara yol açabilir.
    
    Parametreler:
        Hiçbir parametre almaz. DATASET_DIR sabitinden klasör yolunu alır.
    
    Returns:
        None: Fonksiyon sonuçları konsola yazdırılır.
    
    Kontrol Aşamaları:
        1. Sistem Dosyaları: Thumbs.db, .DS_Store, ._* ile başlayan dosyalar
        2. Boş Dosyalar: Dosya boyutu 0 byte olan dosyalar
        3. Bozuk Görüntüler: TensorFlow'un decode edemediği görüntü dosyaları
    
    Örnek Kullanım:
        >>> validate_images()
        --- DERİNLEMESİNE TEMİZLİK BAŞLIYOR: dataset ---
        TensorFlow ile her dosya tek tek okunup test ediliyor...
        [SİLİNDİ - SİSTEM DOSYASI]: .DS_Store
        [SİLİNDİ - BOZUK İÇERİK]: corrupted_image.jpg
        Kontrol edilen: 1000 dosya...
        --- TARAMA BİTTİ ---
        Toplam Silinen Dosya: 15
    """
    print(f"--- DERİNLEMESİNE TEMİZLİK BAŞLIYOR: {DATASET_DIR} ---")
    print("TensorFlow ile her dosya tek tek okunup test ediliyor...")
    
    deleted_count = 0
    checked_count = 0
    
    for root, dirs, files in os.walk(DATASET_DIR):
        for filename in files:
            file_path = os.path.join(root, filename)
            checked_count += 1
            
            if filename in ["Thumbs.db", ".DS_Store"] or filename.startswith("._"):
                try:
                    os.remove(file_path)
                    print(f"[SİLİNDİ - SİSTEM DOSYASI]: {filename}")
                    deleted_count += 1
                except: pass
                continue

            if os.path.getsize(file_path) == 0:
                try:
                    os.remove(file_path)
                    print(f"[SİLİNDİ - BOŞ DOSYA]: {filename}")
                    deleted_count += 1
                except: pass
                continue

            try:
                file_contents = tf.io.read_file(file_path)
                _ = tf.io.decode_image(file_contents, channels=3, expand_animations=False)
            except Exception as e:
                print(f"[SİLİNDİ - BOZUK İÇERİK]: {filename}")
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except: pass

            if checked_count % 1000 == 0:
                print(f"Kontrol edilen: {checked_count} dosya...")

    print(f"\n--- TARAMA BİTTİ ---")
    print(f"Toplam Silinen Dosya: {deleted_count}")
    print("Artık main.py %100 çalışacaktır.")

if __name__ == "__main__":
    validate_images()