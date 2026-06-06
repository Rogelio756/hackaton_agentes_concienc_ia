# FoodSense CDMX 💧
**Sistema de predicción de riesgos de inundación urbana — Ciudad de México**

> Hackathon ConciencIA · Equipo AquaInfer · 2026

## 🚀 API en producción
```
https://hackatonagentesconciencia-production.up.railway.app/predict
https://hackatonagentesconciencia-production.up.railway.app/health
https://hackatonagentesconciencia-production.up.railway.app/zonas
```

---

## ¿Qué es?

Sistema multi-agente que predice en tiempo real el riesgo de inundación para las 16 alcaldías de la CDMX, combinando datos climáticos en vivo con modelos de machine learning entrenados sobre 150 años de datos históricos de CONAGUA.

---

## Arquitectura

```
Open-Meteo API (clima en vivo)
        ↓
  Agente 1 — XGBoost          → ¿Lloverá fuerte mañana? (prob. 0-1)
  Agente 2 — XGBoost Regressor → Nivel de riesgo de inundación (1-5)
  Agente 3 — LLM (Watsonx/Claude) → Texto de alerta para ciudadanos
        ↓
  FastAPI  →  GET /predict  →  Frontend
```

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/predict` | JSON con las 16 alcaldías (actualiza cada 5 min) |
| GET | `/predict/{zona}` | Predicción para una zona específica |
| GET | `/health` | Health check |
| GET | `/` | Dashboard HTML |

### Formato de respuesta `/predict`

```json
{
  "timestamp": "2026-06-06T14:30:00Z",
  "zones": {
    "Iztapalapa": {
      "lluvia_mm": 11.1,
      "prob_lluvia": 0.85,
      "nivel_riesgo": 4,
      "alerta": "Alerta en Iztapalapa. Lluvia de 11.1mm hoy..."
    }
  }
}
```

### Zonas disponibles
`Iztapalapa`, `G.A. Madero`, `Álvaro Obregón`, `Cuauhtémoc`, `Tlalpan`,
`Coyoacán`, `Iztacalco`, `Venustiano Carranza`, `Azcapotzalco`, `Xochimilco`,
`Tláhuac`, `Benito Juárez`, `Miguel Hidalgo`, `Cuajimalpa`,
`Magdalena Contreras`, `Milpa Alta`

---

## Deploy en Railway

### 1. Clonar el repo
```bash
git clone https://github.com/tu-org/aquainfer-cdmx
cd aquainfer-cdmx
```

### 2. Crear proyecto en Railway
```bash
railway login
railway init
railway up
```

### 3. Variables de entorno (opcionales — mejoran el texto de alerta)
En Railway → Settings → Variables:

```
WATSONX_API_KEY=tu_api_key        # IBM Watsonx (recomendado para pitch)
WATSONX_PROJECT_ID=tu_project_id
ANTHROPIC_API_KEY=tu_api_key      # Fallback Claude
INTERVALO_SEG=300                 # Actualización en segundos (default: 300)
```

> Sin API keys funciona igual — el Agente 3 genera alertas por reglas automáticamente.

---

## Correr localmente

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

Abrir: http://localhost:8000/predict

---

## Datos y modelos

| Componente | Detalle |
|------------|---------|
| Dataset base | CONAGUA/SIH — 591,706 registros diarios (1877–2024) |
| Atlas de riesgo | SPCGIR/CONAGUA 2019 — 16 alcaldías × nivel 1-5 |
| Agente 1 | XGBoost · ROC-AUC: 0.8595 · Recall: 79% |
| Agente 2 | XGBoost Regressor · RMSE: 0.011 · R²: 0.9999 |
| Agente 3 | IBM Watsonx Granite / Claude Haiku / Reglas |
| Clima en vivo | Open-Meteo API (gratuita, sin clave) |

---

## Equipo

Hackathon ConciencIA 2026 — Equipo FoodSense CDMX
