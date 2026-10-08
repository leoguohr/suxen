from pathlib import Path
p=Path(__file__).resolve().parent/'only804_mu/effective_run.py'
exec(compile(p.read_text(),str(p),'exec'),{'__file__':str(p),'__name__':'__main__'})
