import os

# Veri seti klasörünün adı
DATASET_DIR = "dataset"

def clean_ghost_files():
    print(f"--- Temizlik Başlıyor: {DATASET_DIR} ---")
    deleted_count = 0
    
    # Klasörleri gez
    for root, dirs, files in os.walk(DATASET_DIR):
        for filename in files:
            # Sadece ._ ile başlayan dosyaları hedef al
            if filename.startswith("._"):
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    # Çok fazla satır kirliliği olmasın diye her 100 dosyada bir yazdır
                    if deleted_count % 100 == 0:
                        print(f"{deleted_count} adet hayalet dosya silindi...")
                except Exception as e:
                    print(f"Silinemedi: {file_path} | Hata: {e}")

    print(f"\n--- İŞLEM TAMAM ---")
    print(f"Toplam Silinen '._' Dosyası: {deleted_count}")
    print("Artık main.py dosyasını çalıştırabilirsiniz.")

if __name__ == "__main__":
    if os.path.exists(DATASET_DIR):
        clean_ghost_files()
    else:
        print(f"HATA: '{DATASET_DIR}' klasörü bulunamadı!")