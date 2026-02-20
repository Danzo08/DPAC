# Platforma Laborator Cybersecurity (Studenti)

## Descriere scurta
Acest proiect este suportul practic pentru materia de **Cybersecurity**, cu focus pe:
- analiza traficului de retea (PCAP),
- pregatirea datelor pentru Machine Learning,
- antrenarea si evaluarea modelelor IDS (Intrusion Detection System).

Scopul este sa parcurgi pipeline-ul complet: **trafic brut -> features -> modele -> evaluare comparativa**.

## Setup rapid (Conda + requirements)
Este recomandat sa folosesti **Python 3.11** (important pentru compatibilitatea TensorFlow).

1. Creeaza mediul Conda:
```bash
conda create -n cyber_lab python=3.11 -y
```

2. Activeaza mediul:
```bash
conda activate cyber_lab
```

3. Instaleaza dependintele:
```bash
pip install -r requirements.txt
```

## Structura folderului
```text
cyber_git_studenti_2026/
├── PLATFORMA_LABORATOR.docx          # ghidul principal al laboratorului
├── README.md                         # instructiuni rapide pentru rulare
├── requirements.txt                  # dependinte Python
├── data/
│   └── pcap_samples/                 # capturi PCAP + CSV-uri packet/flow
├── KDDTest+.arff/
│   └── KDDTest+.arff                 # dataset ARFF pentru exercitii
└── solutii/
    ├── generate_pcap.py
    ├── lab2_pcap_processing.py
    ├── lab2b_ids_intro.ipynb
    ├── lab3_data_preparation.py
    ├── lab3_data_preparation.ipynb
    ├── lab4_classical_ml.py
    ├── lab4_classical_ml.ipynb
    ├── lab5_deep_learning.py
    ├── lab5_deep_learning.ipynb
    ├── lab6_evaluation.py
    ├── lab6_evaluation.ipynb
    └── lab3_out_train/               # exemplu de date preprocesate
```

## Rulare in ordinea laboratoarelor
Ruleaza comenzile din radacina proiectului (`cyber_git_studenti_2026`).

1. **Lab 1/2 - Generare date PCAP**
```bash
python solutii/generate_pcap.py --all --output-dir data/pcap_samples
```

2. **Lab 2 - Procesare PCAP (pachete/flow-uri)**
```bash
python solutii/lab2_pcap_processing.py "data/pcap_samples/http_traffic.pcap"
```

3. **Lab 3 - Pregatirea datelor pentru ML**
```bash
python solutii/lab3_data_preparation.py --input "KDDTrain+.txt" --outdir "solutii/lab3_out_train" --scale standard --test_size 0.2
```
Nota: daca nu ai `KDDTrain+.txt`, scriptul poate genera dataset sintetic (fara `--input`).

4. **Lab 4 - Machine Learning clasic**
```bash
python solutii/lab4_classical_ml.py --data_dir "solutii/lab3_out_train" --outdir "solutii/lab4_out"
```

5. **Lab 5 - Deep Learning (MLP + LSTM)**
```bash
python solutii/lab5_deep_learning.py --data_dir "solutii/lab3_out_train" --outdir "solutii/lab5_out"
```

6. **Lab 6 - Evaluare finala si comparatie modele**
```bash
python solutii/lab6_evaluation.py --data_dir "solutii/lab3_out_train" --ml_models_dir "solutii/lab4_out" --dl_models_dir "solutii/lab5_out" --outdir "solutii/lab6_out"
```

## Descriere scurta pentru fiecare cod (in ordinea platformei)
- `solutii/generate_pcap.py`: genereaza scenarii de trafic (`normal`, `http_traffic`, `portscan`, `synflood`) in format PCAP.
- `solutii/lab2_pcap_processing.py`: citeste PCAP, extrage feature-uri la nivel de pachet/flow si poate exporta CSV.
- `solutii/lab2b_ids_intro.ipynb`: introducere practica in IDS/Snort (format notebook).
- `solutii/lab3_data_preparation.py`: curata/preproceseaza date NSL-KDD, encode + scaling + split train/test, salveaza `.npy/.csv/.pkl`.
- `solutii/lab3_data_preparation.ipynb`: varianta notebook pentru acelasi flux din Lab 3.
- `solutii/lab4_classical_ml.py`: antreneaza si compara Decision Tree, Random Forest, KNN; salveaza modele + grafice + rezultate.
- `solutii/lab4_classical_ml.ipynb`: varianta notebook pentru Lab 4.
- `solutii/lab5_deep_learning.py`: antreneaza MLP si LSTM pe datele din Lab 3, salveaza modele, curbe de invatare si metrici.
- `solutii/lab5_deep_learning.ipynb`: varianta notebook pentru Lab 5.
- `solutii/lab6_evaluation.py`: incarca modelele din Lab 4/5 si produce comparatie finala (metrici, ROC, FP/FN, matrici de confuzie).
- `solutii/lab6_evaluation.ipynb`: varianta notebook pentru Lab 6.

## Observatii utile
- Pentru scripturile care genereaza grafice in terminal/headless, fisierele sunt salvate automat in folderul `--outdir`.
- Daca nu exista unele fisiere `.npy` sau modele, scripturile Lab 5/6 pot folosi fallback pe date sintetice pentru demo.
