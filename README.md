## Instalacija

Neophodno je imati Python verziju 3.10 ili višu.

U željenom okruženju, pokrenuti

```bash
pip install -r requirements.txt
```

## Neuronske mreže

Relevantni fajlovi za predmet Neuronske mreže nalaze se na sledećim putanjama

- models/NN/train.py (trening za obične LSTM i GRU modele)
- models/DL/conv_lstm.ipynb (model koji dodaje conv sloj pre RNN-a)
- models/DL/transp_conv_lstm.ipynb (model koji dodaje conv sloj pre RNN-a, i transponovanu konvoluciju nakon)

## Podaci
Podaci se nalaze u train_belgrade.csv fajlu. Skripte koje ih prikupljaju i obrađuju su

- ./etl.py
- ./detect.py

## Autori

- [@pavleglusac](https://github.com/pavleglusac) R2 15/2023
- [@nevenaradesic](https://github.com/NevenaRadesic) R2 2/2023
