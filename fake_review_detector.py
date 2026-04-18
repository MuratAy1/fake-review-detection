"""
============================================================================
SAHTE YORUM TESPİTİ - MAKİNE ÖĞRENMESİ PROJESİ
Fake Reviews and Opinion Spam Detector Using ML Algorithms
============================================================================
Lisans Bitirme Tezi - İskelet Kod (Skeleton Code)

AÇIKLAMA:
    Bu script, sahte ve gerçek yorumları sınıflandırmak için 4 farklı
    makine öğrenmesi algoritmasını karşılaştırır:
    1. Naive Bayes (baseline - en basit model)
    2. Logistic Regression (doğrusal sınıflandırıcı)
    3. Support Vector Machine - SVM (güçlü metin sınıflandırıcı)
    4. Random Forest (ağaç tabanlı topluluk modeli)

VERİ SETİ:
    Kaggle "Fake Reviews Dataset"
    - 20,000 sahte (CG - Computer Generated) yorum
    - 20,000 gerçek (OR - Original Review) yorum
    İndirme: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset

KULLANIM:
    1. Gerekli kütüphaneleri kur: pip install pandas scikit-learn matplotlib seaborn nltk wordcloud
    2. Veri setini indir ve bu script ile aynı klasöre koy
    3. Çalıştır: python fake_review_detector.py

YAZAR: [Senin İsmin]
TARİH: [Tarih]
============================================================================
"""

# ==========================================================================
# BÖLÜM 1: KÜTÜPHANE İMPORTLARI
# ==========================================================================
# Her kütüphanenin ne işe yaradığını anla — tez savunmasında sorabilirler!

import pandas as pd                  # Veri okuma, temizleme, manipülasyon
import numpy as np                   # Matematiksel işlemler, diziler
import matplotlib.pyplot as plt      # Grafik çizimi
import seaborn as sns                # Daha güzel istatistiksel grafikler
import re                            # Regex - metin temizleme için
import string                        # Noktalama işaretleri listesi
import os                            # Dosya yolu işlemleri
import warnings                      # Uyarıları gizleme
warnings.filterwarnings('ignore')

# ----- Doğal Dil İşleme (NLP) Kütüphaneleri -----
import nltk                          # Natural Language Toolkit
nltk.download('stopwords', quiet=True)   # "the", "is", "at" gibi anlamsız kelimeler
nltk.download('punkt', quiet=True)       # Cümle/kelime ayırıcı
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)     # Kelime kökleri veritabanı
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ----- Scikit-learn: Makine Öğrenmesi Kütüphanesi -----
from sklearn.model_selection import train_test_split    # Veriyi eğitim/test olarak böl
from sklearn.feature_extraction.text import TfidfVectorizer  # Metni sayılara çevir
from sklearn.naive_bayes import MultinomialNB           # Naive Bayes algoritması
from sklearn.linear_model import LogisticRegression     # Lojistik Regresyon
from sklearn.svm import LinearSVC                       # Support Vector Machine
from sklearn.ensemble import RandomForestClassifier     # Random Forest
from sklearn.metrics import (                           # Değerlendirme metrikleri
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.model_selection import cross_val_score     # Çapraz doğrulama

# Wordcloud (opsiyonel - güzel görselleştirme için)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("WordCloud kurulu değil. Kurulum: pip install wordcloud")


# ==========================================================================
# BÖLÜM 2: VERİ YÜKLEME VE İLK KEŞİF (Exploratory Data Analysis - EDA)
# ==========================================================================

def load_data(filepath):
    """
    Veri setini yükle ve temel bilgileri göster.

    TEZE NOT: Bu bölümde veri setinin özelliklerini açıkla:
    - Kaç satır/sütun var?
    - Sınıf dağılımı nasıl? (dengeli mi?)
    - Eksik veri var mı?
    """
    print("=" * 60)
    print("ADIM 1: VERİ YÜKLEME")
    print("=" * 60)

    # CSV dosyasını oku
    df = pd.read_csv(filepath)

    # Temel bilgiler
    print(f"\n📊 Veri seti boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
    print(f"\n📋 Sütunlar: {list(df.columns)}")
    print(f"\n🔍 İlk 5 satır:")
    print(df.head())
    print(f"\n📈 Veri tipleri:")
    print(df.dtypes)
    print(f"\n❓ Eksik veri sayısı:")
    print(df.isnull().sum())

    return df


def explore_data(df, text_column, label_column):
    """
    Keşifsel Veri Analizi (EDA) — tez raporunun en önemli bölümlerinden biri.

    Bu fonksiyon şunları üretir:
    1. Sınıf dağılımı grafiği (fake vs real)
    2. Yorum uzunluğu dağılımı
    3. Kelime bulutu (word cloud)
    4. En sık kullanılan kelimeler
    """
    print("\n" + "=" * 60)
    print("ADIM 2: KEŞİFSEL VERİ ANALİZİ (EDA)")
    print("=" * 60)

    # --- 2a: Sınıf Dağılımı ---
    print(f"\n🏷️  Sınıf dağılımı:")
    print(df[label_column].value_counts())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Keşifsel Veri Analizi (EDA)', fontsize=16, fontweight='bold')

    # Grafik 1: Sınıf dağılımı bar chart
    class_counts = df[label_column].value_counts()
    colors = ['#2ecc71', '#e74c3c']  # yeşil = gerçek, kırmızı = sahte
    axes[0, 0].bar(class_counts.index, class_counts.values, color=colors)
    axes[0, 0].set_title('Sınıf Dağılımı (Fake vs Real)')
    axes[0, 0].set_xlabel('Etiket')
    axes[0, 0].set_ylabel('Yorum Sayısı')
    for i, v in enumerate(class_counts.values):
        axes[0, 0].text(i, v + 200, str(v), ha='center', fontweight='bold')

    # --- 2b: Yorum uzunluğu analizi ---
    df['review_length'] = df[text_column].astype(str).apply(len)
    df['word_count'] = df[text_column].astype(str).apply(lambda x: len(x.split()))

    # Grafik 2: Yorum uzunluğu dağılımı (karakter)
    for label in df[label_column].unique():
        subset = df[df[label_column] == label]
        axes[0, 1].hist(subset['review_length'], bins=50, alpha=0.6, label=label)
    axes[0, 1].set_title('Yorum Uzunluğu Dağılımı (Karakter)')
    axes[0, 1].set_xlabel('Karakter Sayısı')
    axes[0, 1].set_ylabel('Frekans')
    axes[0, 1].legend()

    # Grafik 3: Kelime sayısı dağılımı
    for label in df[label_column].unique():
        subset = df[df[label_column] == label]
        axes[1, 0].hist(subset['word_count'], bins=50, alpha=0.6, label=label)
    axes[1, 0].set_title('Kelime Sayısı Dağılımı')
    axes[1, 0].set_xlabel('Kelime Sayısı')
    axes[1, 0].set_ylabel('Frekans')
    axes[1, 0].legend()

    # Grafik 4: Ortalama istatistikler
    stats = df.groupby(label_column)['word_count'].mean()
    axes[1, 1].bar(stats.index, stats.values, color=colors)
    axes[1, 1].set_title('Ortalama Kelime Sayısı')
    axes[1, 1].set_xlabel('Etiket')
    axes[1, 1].set_ylabel('Ortalama Kelime Sayısı')

    plt.tight_layout()
    plt.savefig('01_eda_analizi.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ EDA grafikleri '01_eda_analizi.png' olarak kaydedildi.")

    # --- 2c: Kelime Bulutu (Word Cloud) ---
    if WORDCLOUD_AVAILABLE:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for idx, label in enumerate(df[label_column].unique()):
            text = ' '.join(df[df[label_column] == label][text_column].astype(str).tolist())
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                max_words=100,
                colormap='viridis' if idx == 0 else 'magma'
            ).generate(text)
            axes[idx].imshow(wordcloud, interpolation='bilinear')
            axes[idx].set_title(f'Kelime Bulutu: {label}', fontsize=14)
            axes[idx].axis('off')
        plt.tight_layout()
        plt.savefig('02_kelime_bulutu.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("✅ Kelime bulutları '02_kelime_bulutu.png' olarak kaydedildi.")

    return df


# ==========================================================================
# BÖLÜM 3: METİN ÖN İŞLEME (Text Preprocessing)
# ==========================================================================
# TEZE NOT: Bu bölüm kritik — hangi adımları neden yaptığını açıkla!

def preprocess_text(text):
    """
    Tek bir metin parçasını temizle ve normalize et.

    Adımlar:
    1. Küçük harfe çevir   → "GREAT Product" → "great product"
    2. URL'leri kaldır      → "check http://..." → "check"
    3. Sayıları kaldır       → "5 stars" → "stars"
    4. Noktalama kaldır      → "great!" → "great"
    5. Stopwords kaldır      → "this is a great" → "great"
    6. Lemmatizasyon          → "running" → "run", "better" → "good"

    NEDEN BU ADIMLAR?
    - ML modelleri ham metni anlayamaz, sayılarla çalışır
    - Gereksiz kelimeleri (the, is, at) kaldırmak sinyal/gürültü oranını artırır
    - Lemmatizasyon farklı çekimleri aynı köke indirger → model daha iyi geneller
    """
    # Boş/NaN kontrolü
    if pd.isna(text):
        return ""

    text = str(text)

    # 1. Küçük harf
    text = text.lower()

    # 2. URL'leri kaldır
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. HTML etiketlerini kaldır
    text = re.sub(r'<.*?>', '', text)

    # 4. Sayıları kaldır
    text = re.sub(r'\d+', '', text)

    # 5. Noktalama işaretlerini kaldır
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 6. Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. Tokenize (kelime kelime ayır)
    tokens = word_tokenize(text)

    # 8. Stopwords kaldır (İngilizce)
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]

    # 9. Lemmatizasyon (kelime köklerini bul)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # 10. Çok kısa kelimeleri kaldır (1 karakter)
    tokens = [t for t in tokens if len(t) > 1]

    return ' '.join(tokens)


def preprocess_dataframe(df, text_column, label_column):
    """
    Tüm veri setine ön işleme uygula.
    """
    print("\n" + "=" * 60)
    print("ADIM 3: METİN ÖN İŞLEME")
    print("=" * 60)

    print("⏳ Metin temizleme başladı... (bu birkaç dakika sürebilir)")
    df['cleaned_text'] = df[text_column].apply(preprocess_text)

    # Temizleme sonrası boş kalan satırları kaldır
    empty_count = (df['cleaned_text'] == '').sum()
    if empty_count > 0:
        print(f"⚠️  {empty_count} boş satır temizleme sonrası kaldırıldı.")
        df = df[df['cleaned_text'] != ''].reset_index(drop=True)

    print(f"✅ Temizleme tamamlandı. Kalan satır: {len(df)}")

    # Önce/sonra karşılaştırma örneği
    print(f"\n📝 Önce/Sonra Örneği:")
    sample_idx = df.index[0]
    print(f"   HAM:   {df[text_column].iloc[0][:100]}...")
    print(f"   TEMİZ: {df['cleaned_text'].iloc[0][:100]}...")

    return df


# ==========================================================================
# BÖLÜM 4: ÖZELLİK ÇIKARIMI (Feature Extraction - TF-IDF)
# ==========================================================================

def extract_features(df, label_column, max_features=10000):
    """
    TF-IDF ile metni sayısal özelliklere dönüştür.

    TF-IDF NEDİR? (Tez savunmasında %100 sorulur!)
    ─────────────────────────────────────────────────
    TF  = Term Frequency  = Bir kelimenin o dokümanda kaç kez geçtiği
    IDF = Inverse Document Frequency = Kelimenin tüm dokümanlardaki nadirliği

    TF-IDF = TF × IDF

    Mantık: "hotel" kelimesi her yorumda geçiyorsa önemli değil (düşük IDF).
    Ama "disgusting" sadece birkaç yorumda geçiyorsa çok bilgilendirici (yüksek IDF).

    max_features=10000: En önemli 10.000 kelimeyi kullan (bellek ve hız için).

    PARAMETRE AÇIKLAMALARI:
    - ngram_range=(1,2): Hem tekli kelimeler hem de ikili kelime çiftleri
      Örnek: "great" (unigram) + "great hotel" (bigram)
    - max_df=0.95: Yorumların %95'inden fazlasında geçen kelimeleri çıkar
    - min_df=2: En az 2 yorumda geçen kelimeleri al
    - sublinear_tf=True: TF'ye logaritmik ölçekleme uygula (büyük frekansları bastırır)
    """
    print("\n" + "=" * 60)
    print("ADIM 4: ÖZELLİK ÇIKARIMI (TF-IDF)")
    print("=" * 60)

    # Etiketleri sayıya çevir (modeller sayılarla çalışır)
    # CG (Computer Generated / Sahte) → 1
    # OR (Original Review / Gerçek)   → 0
    label_map = {'CG': 1, 'OR': 0}
    # Eğer veri setindeki etiketler farklıysa bu mapping'i güncelle!
    if df[label_column].dtype == 'object':
        unique_labels = df[label_column].unique()
        print(f"   Bulunan etiketler: {unique_labels}")
        # Otomatik mapping: ilk etiket → 0, ikinci → 1
        if 'CG' in unique_labels and 'OR' in unique_labels:
            df['label_numeric'] = df[label_column].map(label_map)
        else:
            # Genel durum: etiketleri otomatik numaralandır
            label_map = {label: idx for idx, label in enumerate(unique_labels)}
            df['label_numeric'] = df[label_column].map(label_map)
        print(f"   Etiket mapping: {label_map}")
    else:
        df['label_numeric'] = df[label_column]

    # Eğitim/Test ayrımı (%80 eğitim, %20 test)
    # stratify: her iki sette de aynı oranda fake/real olmasını sağlar
    # random_state=42: tekrarlanabilir sonuçlar (tez için önemli!)
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'],
        df['label_numeric'],
        test_size=0.20,
        random_state=42,
        stratify=df['label_numeric']
    )

    print(f"\n📊 Eğitim seti: {len(X_train)} yorum")
    print(f"📊 Test seti:   {len(X_test)} yorum")

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        max_features=max_features,  # En önemli N kelime
        ngram_range=(1, 2),         # Unigram + Bigram
        max_df=0.95,                # Çok yaygın kelimeleri çıkar
        min_df=2,                   # Çok nadir kelimeleri çıkar
        sublinear_tf=True           # Logaritmik TF ölçekleme
    )

    # Eğitim setinden öğren (fit) ve dönüştür (transform)
    X_train_tfidf = tfidf.fit_transform(X_train)
    # Test setini sadece dönüştür (transform) — fit ETME! (veri sızıntısı olur)
    X_test_tfidf = tfidf.transform(X_test)

    print(f"\n🔢 TF-IDF matris boyutu: {X_train_tfidf.shape}")
    print(f"   ({X_train_tfidf.shape[0]} yorum × {X_train_tfidf.shape[1]} özellik)")

    # En önemli kelimeleri göster
    feature_names = tfidf.get_feature_names_out()
    print(f"\n📝 Örnek özellikler (ilk 20): {list(feature_names[:20])}")

    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf, label_map


# ==========================================================================
# BÖLÜM 5: MODEL EĞİTİMİ VE DEĞERLENDİRME
# ==========================================================================

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    4 farklı ML algoritmasını eğit, test et ve karşılaştır.

    TEZE NOT: Her algoritmanın nasıl çalıştığını kısaca açıkla:

    1. NAIVE BAYES (MultinomialNB):
       - Bayes teoremine dayanır: P(Sahte|kelimeler) hesaplar
       - "Naive" çünkü kelimelerin birbirinden bağımsız olduğunu varsayar
       - Metin sınıflandırmada çok hızlı ve şaşırtıcı derecede etkili
       - Baseline (temel karşılaştırma modeli) olarak kullanılır

    2. LOGISTIC REGRESSION:
       - Doğrusal bir sınıflandırıcı (lineer karar sınırı çizer)
       - Sigmoid fonksiyonu ile olasılık çıktısı verir
       - Yüksek boyutlu metin verilerinde çok iyi çalışır
       - max_iter=1000: optimizasyonun yakınsaması için yeterli iterasyon

    3. SUPPORT VECTOR MACHINE (LinearSVC):
       - İki sınıf arasında en geniş boşluğu (margin) bırakacak
         hiper-düzlemi (hyperplane) bulur
       - Metin sınıflandırmada genellikle en iyi performansı verir
       - LinearSVC: büyük veri setleri için optimize edilmiş SVM

    4. RANDOM FOREST:
       - Birçok karar ağacının "oy" vermesiyle sınıflandırma yapar
       - n_estimators=200: 200 adet karar ağacı oluşturur
       - Overfitting'e (aşırı öğrenme) karşı dirençlidir
       - Metin verisinde genellikle SVM'den düşük performans gösterir
         (ama farklılığı göstermek tez için değerli!)
    """
    print("\n" + "=" * 60)
    print("ADIM 5: MODEL EĞİTİMİ VE DEĞERLENDİRME")
    print("=" * 60)

    # Model sözlüğü
    models = {
        'Naive Bayes': MultinomialNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
        'SVM (LinearSVC)': LinearSVC(max_iter=2000, random_state=42, C=1.0),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    }

    # Sonuçları sakla
    results = {}

    for name, model in models.items():
        print(f"\n{'─' * 40}")
        print(f"🤖 {name} eğitiliyor...")

        # Modeli eğit
        model.fit(X_train, y_train)

        # Test seti üzerinde tahmin yap
        y_pred = model.predict(X_test)

        # Metrikleri hesapla
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Çapraz doğrulama (5-fold) — modelin güvenilirliğini test eder
        # Veriyi 5 parçaya böler, her seferinde 1 parçayı test olarak kullanır
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        results[name] = {
            'model': model,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred
        }

        print(f"   ✅ Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
        print(f"   ✅ Precision: {prec:.4f}")
        print(f"   ✅ Recall:    {rec:.4f}")
        print(f"   ✅ F1 Score:  {f1:.4f}")
        print(f"   ✅ CV Score:  {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        print(f"\n   📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Gerçek (OR)', 'Sahte (CG)']))

    return results


# ==========================================================================
# BÖLÜM 6: GÖRSELLEŞTİRME (Tez için Grafikler)
# ==========================================================================

def plot_results(results, y_test):
    """
    Tez raporu için profesyonel grafikler üret:
    1. Model karşılaştırma tablosu
    2. Confusion matrix (karışıklık matrisi) her model için
    3. Metrik karşılaştırma bar chart
    """
    print("\n" + "=" * 60)
    print("ADIM 6: GÖRSELLEŞTİRME")
    print("=" * 60)

    model_names = list(results.keys())

    # --- 6a: Karışıklık Matrisleri (Confusion Matrices) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Karışıklık Matrisleri (Confusion Matrices)', fontsize=16, fontweight='bold')

    for idx, (name, data) in enumerate(results.items()):
        ax = axes[idx // 2, idx % 2]
        cm = confusion_matrix(y_test, data['y_pred'])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Gerçek (OR)', 'Sahte (CG)'],
            yticklabels=['Gerçek (OR)', 'Sahte (CG)'],
            ax=ax
        )
        ax.set_title(f'{name}\n(Accuracy: {data["accuracy"]:.4f})')
        ax.set_xlabel('Tahmin Edilen')
        ax.set_ylabel('Gerçek Değer')

    plt.tight_layout()
    plt.savefig('03_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Confusion matrisleri '03_confusion_matrices.png' olarak kaydedildi.")

    # --- 6b: Metrik Karşılaştırma Bar Chart ---
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.2
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[name][metric] for name in model_names]
        bars = ax.bar(x + i * width, values, width, label=label, color=colors[i])
        # Değerleri barların üstüne yaz
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Model')
    ax.set_ylabel('Skor')
    ax.set_title('Model Performans Karşılaştırması', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('04_model_karsilastirma.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Model karşılaştırması '04_model_karsilastirma.png' olarak kaydedildi.")

    # --- 6c: Çapraz Doğrulama (Cross-Validation) Sonuçları ---
    fig, ax = plt.subplots(figsize=(10, 6))
    cv_means = [results[name]['cv_mean'] for name in model_names]
    cv_stds = [results[name]['cv_std'] for name in model_names]

    bars = ax.bar(model_names, cv_means, yerr=cv_stds, capsize=5,
                  color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
                  edgecolor='black', linewidth=0.5)
    for bar, mean, std in zip(bars, cv_means, cv_stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.005,
                f'{mean:.4f}\n±{std:.4f}', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Accuracy')
    ax.set_title('5-Fold Çapraz Doğrulama Sonuçları', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('05_cross_validation.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ CV sonuçları '05_cross_validation.png' olarak kaydedildi.")


def generate_summary_table(results):
    """
    Tez raporuna kopyalanabilecek özet tablo oluştur.
    """
    print("\n" + "=" * 60)
    print("SONUÇ TABLOSU (Teze kopyalayabilirsin)")
    print("=" * 60)

    summary = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [f"{results[n]['accuracy']:.4f}" for n in results],
        'Precision': [f"{results[n]['precision']:.4f}" for n in results],
        'Recall': [f"{results[n]['recall']:.4f}" for n in results],
        'F1 Score': [f"{results[n]['f1_score']:.4f}" for n in results],
        'CV Mean': [f"{results[n]['cv_mean']:.4f}" for n in results],
        'CV Std': [f"±{results[n]['cv_std']:.4f}" for n in results],
    })

    print(summary.to_string(index=False))
    summary.to_csv('06_sonuc_tablosu.csv', index=False)
    print("\n✅ Sonuç tablosu '06_sonuc_tablosu.csv' olarak kaydedildi.")

    # En iyi modeli belirle
    best_model = max(results, key=lambda x: results[x]['f1_score'])
    print(f"\n🏆 EN İYİ MODEL: {best_model}")
    print(f"   F1 Score: {results[best_model]['f1_score']:.4f}")
    print(f"   Accuracy: {results[best_model]['accuracy']:.4f}")

    return summary


# ==========================================================================
# BÖLÜM 7: ANA ÇALIŞTIRMA (Main Pipeline)
# ==========================================================================

def main():
    """
    Tüm pipeline'ı sırayla çalıştır.

    KULLANMADAN ÖNCE:
    1. Veri setini indir: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset
    2. CSV dosyasının yolunu aşağıda güncelle
    3. Sütun isimlerini kontrol et ve gerekirse güncelle
    """

    # ╔══════════════════════════════════════════════════════════╗
    # ║  ⚠️ BU ALANI KENDİ VERİ SETİNE GÖRE GÜNCELLE!         ║
    # ╚══════════════════════════════════════════════════════════╝

    DATA_FILE = 'fake_reviews_dataset.csv'    # Veri seti dosya yolu
    TEXT_COLUMN = 'text_'                     # Yorum metni sütunu (veri setine göre değiştir!)
    LABEL_COLUMN = 'label'                   # Etiket sütunu (CG/OR)

    # Dosya kontrolü
    if not os.path.exists(DATA_FILE):
        print(f"❌ HATA: '{DATA_FILE}' dosyası bulunamadı!")
        print(f"   Lütfen veri setini indirip bu klasöre koyun.")
        print(f"   İndirme: https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset")
        print(f"\n   Alternatif: Dosya yolunu DATA_FILE değişkeninde güncelleyin.")
        return

    # Pipeline'ı çalıştır
    print("🚀 Sahte Yorum Tespit Pipeline'ı Başlatılıyor...\n")

    # 1. Veri yükle
    df = load_data(DATA_FILE)

    # 2. Keşifsel veri analizi
    df = explore_data(df, TEXT_COLUMN, LABEL_COLUMN)

    # 3. Metin ön işleme
    df = preprocess_dataframe(df, TEXT_COLUMN, LABEL_COLUMN)

    # 4. Özellik çıkarımı (TF-IDF)
    X_train, X_test, y_train, y_test, tfidf, label_map = extract_features(
        df, LABEL_COLUMN, max_features=10000
    )

    # 5. Model eğitimi ve değerlendirme
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # 6. Görselleştirme
    plot_results(results, y_test)

    # 7. Özet tablo
    summary = generate_summary_table(results)

    print("\n" + "=" * 60)
    print("✅ PIPELINE TAMAMLANDI!")
    print("=" * 60)
    print("\n📁 Oluşturulan dosyalar:")
    print("   • 01_eda_analizi.png        - Keşifsel veri analizi grafikleri")
    print("   • 02_kelime_bulutu.png      - Kelime bulutları (word clouds)")
    print("   • 03_confusion_matrices.png - Karışıklık matrisleri")
    print("   • 04_model_karsilastirma.png - Model performans karşılaştırması")
    print("   • 05_cross_validation.png   - Çapraz doğrulama sonuçları")
    print("   • 06_sonuc_tablosu.csv      - Sonuç özet tablosu")
    print("\n💡 İPUCU: Bu grafikleri doğrudan tez raporuna ekleyebilirsin!")


# Script doğrudan çalıştırıldığında main() fonksiyonunu başlat
if __name__ == "__main__":
    main()
