# Agente 1 — Detector de Lluvia Intensa (Forecasting t+1)

## Objetivo
Predice si el dia SIGUIENTE tendra lluvia intensa (>= 10mm),
usando como input las condiciones climaticas del dia actual.

## Modelo
- Algoritmo    : XGBoostClassifier v2 (forecasting)
- Target       : lluvia_intensa_t1 — lluvia >= 10mm manana (1/0)
- ROC-AUC      : 0.8595
- Split        : temporal — train <= 2000 / test > 2000

## Features (input del dia actual)
- MONTH
- DAY
- RAINFALL
- MAXT
- MINT
- rain_3d
- rain_7d
- rain_30d
- temporada_lluvias
- riesgo_precip_max
- riesgo_precip_mean
- alcaldia_encoded

## Archivos
- agente1_xgboost.pkl          — modelo entrenado
- label_encoder_alcaldia.pkl   — encoder de alcaldias
- metricas_agente1.json        — metricas completas

## Uso

```python
import pickle, pandas as pd

with open('agente1_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder_alcaldia.pkl', 'rb') as f:
    le = pickle.load(f)

# Preparar input con datos del dia actual
X_hoy = pd.DataFrame([{
    'MONTH': 7, 'DAY': 15, 'RAINFALL': 5.2,
    'MAXT': 22.0, 'MINT': 14.0,
    'rain_3d': 12.1, 'rain_7d': 28.4, 'rain_30d': 95.0,
    'temporada_lluvias': 1,
    'riesgo_precip_max': 4, 'riesgo_precip_mean': 2.93,
    'alcaldia_encoded': le.transform(['Tlalpan'])[0]
}])

proba = model.predict_proba(X_hoy)[0, 1]
print(f'Probabilidad lluvia intensa manana: {proba:.1%}')
```

Generado: 2026-06-06 — AquaInfer CDMX
