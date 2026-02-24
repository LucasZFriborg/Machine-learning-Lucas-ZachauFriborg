# Python-miljö – VIKTIGT

Detta projekt använder:

- uv
- ett lokalt virtual environment i `.venv/`
- Python-versionen KAN ändras under projektets gång

Miljön ska alltid gå att återskapa från grunden.

---

## Nuvarande läge (första fasen)

- Python-version: 3.13 (tillfällig)
- Huvudbibliotek: scikit-learn, numpy, pandas
- Snapshot av beroenden: `requirements.txt`

`requirements.txt` MÅSTE alltid spegla den faktiska miljön.

---

## REGEL (hoppa inte över)

Efter att du installerar eller uppdaterar ETT ENDA paket, kör alltid:

    uv pip freeze > requirements.txt

Detta är obligatoriskt för att miljön ska kunna återskapas senare.

---

## När PyTorch börjar användas (VIKTIG TRANSITION)

När projektet börjar använda PyTorch MÅSTE Python-versionen bytas
till **Python 3.11** av kompatibilitetsskäl.

Följ stegen NEDAN EXAKT, från projektets rotkatalog:

### 1. Avaktivera och ta bort nuvarande miljö

    deactivate
    rm -rf .venv

### 2. Installera Python 3.11 via uv (om det inte redan är gjort)

    uv python install 3.11

### 3. Skapa nytt virtual environment med Python 3.11

    uv venv --python 3.11

### 4. Installera tillbaka alla tidigare beroenden

    uv pip install -r requirements.txt

### 5. Installera PyTorch

    uv pip install torch torchvision torchaudio

### 6. Verifiera att allt fungerar

    python --version
    # ska visa Python 3.11.x

    python -c "import torch; print(torch.__version__)"

---

## Viktiga noter

- `.venv/` ska INTE committas till git
- `requirements.txt` ska ALLTID committas
- Att radera och återskapa `.venv` är normalt och förväntat
- Försök ALDRIG byta Python-version i ett befintligt venv

Om något strular:  
→ ta bort `.venv` och skapa om den.