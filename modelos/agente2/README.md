# Agente 2 - Clasificador de Riesgo de Inundacion (v2)

## Objetivo
Estima el nivel de riesgo de inundacion (1.0-5.0) de una alcaldia
a partir de las condiciones climaticas del dia actual.
Sin data leakage: NO usa scores de riesgo como features.

## Modelo
- Algoritmo    : XGBoostRegressor (target continuo)
- Target       : riesgo_inund_mean (1.0 = muy bajo, 5.0 = muy alto)
- RMSE         : 0.0110
- MAE          : 0.0073
- R2           : 0.9999
- ROC-AUC bin  : 1.0000  (umbral >= 3.0)
- Split        : temporal train <= 2000 / test > 2000

## Features (solo clima + ubicacion)
- alcaldia_encoded
- MONTH
- DAY
- RAINFALL
- MAXT
- MINT
- rain_3d
- rain_7d
- rain_30d
- temporada_lluvias

## Interpretacion del output
- Salida continua 1.0-5.0: nivel de riesgo estimado
- >= 3.0 = riesgo alto -> activar Agente 3

## Archivos
- agente2_xgboost_regressor.pkl  - modelo entrenado
- label_encoder_alcaldia.pkl     - encoder de alcaldias
- metricas_agente2.json          - metricas completas

## Uso

```python
import pickle, pandas as pd

with open('agente2_xgboost_regressor.pkl', 'rb') as f:
    model = pickle.load(f)
with open('label_encoder_alcaldia.pkl', 'rb') as f:
    le = pickle.load(f)

X_hoy = pd.DataFrame([{
    'alcaldia_encoded': le.transform(['Iztapalapa'])[0],
    'MONTH': 8, 'DAY': 10, 'RAINFALL': 15.0,
    'MAXT': 21.0, 'MINT': 13.0,
    'rain_3d': 32.0, 'rain_7d': 58.0, 'rain_30d': 120.0,
    'temporada_lluvias': 1
}])

riesgo = model.predict(X_hoy)[0]
print(f'Riesgo estimado: {riesgo:.2f} / 5.0')
```

Generado: 2026-06-06 - AquaInfer CDMX
