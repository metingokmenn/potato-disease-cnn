import os

DATASET_DIR = "dataset"

def clean_ghost_files():
    """
    Veri seti klasöründeki macOS hayalet dosyalarını temizler.
    
    Bu fonksiyon, macOS işletim sisteminden Windows'a aktarılan dosyalarda 
    oluşan "._" ile başlayan hayalet dosyaları (ghost files) bulur ve siler.
    Bu dosyalar macOS'un metadata bilgilerini içerir ve Windows'ta gereksizdir.
    Veri seti yükleme sırasında hatalara neden olabilirler.
    
    Parametreler:
        Hiçbir parametre almaz. DATASET_DIR sabitinden klasör yolunu alır.
    
    Returns:
        None: Fonksiyon sonuçları konsola yazdırılır.
    
    İşlem:
        - DATASET_DIR klasörü ve alt klasörlerinde recursive olarak gezinir
        - "._" ile başlayan tüm dosyaları bulur ve siler
        - Her 100 silinen dosyada bir ilerleme mesajı gösterir
    
    Örnek Kullanım:
        >>> clean_ghost_files()
        --- Temizlik Başlıyor: dataset ---
        100 adet hayalet dosya silindi...
        --- İŞLEM TAMAM ---
        Toplam Silinen '._' Dosyası: 245
    """
    print(f"--- Temizlik Başlıyor: {DATASET_DIR} ---")
    deleted_count = 0
    
    for root, dirs, files in os.walk(DATASET_DIR):
        for filename in files:
            if filename.startswith("._"):
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    deleted_count += 1
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