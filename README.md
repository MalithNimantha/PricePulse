## Windows setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Notebook

```powershell
.\.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace notebooks/ensemble_analysis.ipynb
```