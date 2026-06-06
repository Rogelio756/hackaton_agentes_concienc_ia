"""
AquaInfer CDMX — API Principal
FastAPI + Agente 3 integrado
Desplegado en Railway
"""

import os
import json
import pickle
import threading
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Configuración ────────────────────────────────────────
BASE     = Path(__file__).parent
MODELOS  = BASE / 'modelos'
REF_PATH = BASE / 'referencia_alcaldias.csv'
STATE    = BASE / 'live_state_v2.json'
INTERVALO = int(os.environ.get('INTERVALO_SEG', 300))

NAME_MAP = {
    'Iztapalapa':            'Iztapalapa',
    'Gustavo A. Madero':     'G.A. Madero',
    'Alvaro Obregon':        'Álvaro Obregón',
    'Cuauhtemoc':            'Cuauhtémoc',
    'Tlalpan':               'Tlalpan',
    'Coyoacan':              'Coyoacán',
    'Iztacalco':             'Iztacalco',
    'Venustiano Carranza':   'Venustiano Carranza',
    'Azcapotzalco':          'Azcapotzalco',
    'Xochimilco':            'Xochimilco',
    'Tlahuac':               'Tláhuac',
    'Benito Juarez':         'Benito Juárez',
    'Miguel Hidalgo':        'Miguel Hidalgo',
    'Cuajimalpa de Morelos': 'Cuajimalpa',
    'La Magdalena Contreras':'Magdalena Contreras',
    'Milpa Alta':            'Milpa Alta',
}

ALCALDIAS_COORDS = {
    'Iztapalapa':            (19.3557, -99.0591),
    'Gustavo A. Madero':     (19.4840, -99.1130),
    'Alvaro Obregon':        (19.3620, -99.2010),
    'Cuauhtemoc':            (19.4269, -99.1436),
    'Tlalpan':               (19.2924, -99.1731),
    'Coyoacan':              (19.3467, -99.1617),
    'Iztacalco':             (19.3950, -99.0950),
    'Venustiano Carranza':   (19.4310, -99.1040),
    'Azcapotzalco':          (19.4868, -99.1847),
    'Xochimilco':            (19.2569, -99.1043),
    'Tlahuac':               (19.2980, -99.0090),
    'Benito Juarez':         (19.3984, -99.1573),
    'Miguel Hidalgo':        (19.4270, -99.1925),
    'Cuajimalpa de Morelos': (19.3597, -99.2988),
    'La Magdalena Contreras':(19.3230, -99.2440),
    'Milpa Alta':            (19.1926, -99.0231),
}

# ── Cargar modelos ───────────────────────────────────────
print("Cargando modelos...")
with open(MODELOS / 'agente1' / 'agente1_xgboost.pkl', 'rb') as f:
    modelo_a1 = pickle.load(f)
with open(MODELOS / 'agente1' / 'label_encoder_alcaldia.pkl', 'rb') as f:
    le_a1 = pickle.load(f)
with open(MODELOS / 'agente2' / 'agente2_xgboost_regressor.pkl', 'rb') as f:
    modelo_a2 = pickle.load(f)
with open(MODELOS / 'agente2' / 'label_encoder_alcaldia.pkl', 'rb') as f:
    le_a2 = pickle.load(f)
ref = pd.read_csv(REF_PATH)
print("Modelos OK")

FEAT_A1 = ['MONTH','DAY','RAINFALL','MAXT','MINT',
           'rain_3d','rain_7d','rain_30d',
           'temporada_lluvias','riesgo_precip_max',
           'riesgo_precip_mean','alcaldia_encoded']
FEAT_A2 = ['alcaldia_encoded','MONTH','DAY','RAINFALL','MAXT','MINT',
           'rain_3d','rain_7d','rain_30d','temporada_lluvias']


# ── Generación de alertas ────────────────────────────────
def alerta_watsonx(nombre, lluvia_mm, prob, nivel, rain_7d):
    try:
        api_key    = os.environ.get('WATSONX_API_KEY','')
        project_id = os.environ.get('WATSONX_PROJECT_ID','')
        if not api_key or not project_id:
            return None
        token = requests.post(
            'https://iam.cloud.ibm.com/identity/token',
            data={'grant_type':'urn:ibm:params:oauth:grant-type:apikey','apikey':api_key},
            headers={'Content-Type':'application/x-www-form-urlencoded'}
        ).json()['access_token']
        nivel_txt = {1:'muy bajo',2:'bajo',3:'medio',4:'alto',5:'muy alto'}.get(nivel,'medio')
        prompt = (f"Eres AquaInfer CDMX. Genera alerta breve (max 2 oraciones) para {nombre}. "
                  f"Lluvia hoy: {lluvia_mm}mm, acumulado 7d: {rain_7d}mm, "
                  f"prob lluvia intensa manana: {prob*100:.0f}%, riesgo: {nivel}/5 ({nivel_txt}). "
                  f"Directo y util para ciudadanos. Sin markdown.")
        resp = requests.post(
            'https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29',
            headers={'Authorization':f'Bearer {token}','Content-Type':'application/json'},
            json={'model_id':'ibm/granite-13b-chat-v2','input':prompt,
                  'parameters':{'max_new_tokens':120,'temperature':0.5},'project_id':project_id},
            timeout=15
        ).json()
        return resp['results'][0]['generated_text'].strip()
    except Exception:
        return None


def alerta_claude(nombre, lluvia_mm, prob, nivel, rain_7d):
    try:
        import anthropic
        api_key = os.environ.get('ANTHROPIC_API_KEY','')
        if not api_key:
            return None
        client = anthropic.Anthropic(api_key=api_key)
        nivel_txt = {1:'muy bajo',2:'bajo',3:'medio',4:'alto',5:'muy alto'}.get(nivel,'medio')
        msg = client.messages.create(
            model="claude-haiku-4-5", max_tokens=120,
            messages=[{"role":"user","content":(
                f"Eres AquaInfer CDMX. Genera alerta breve (max 2 oraciones) para {nombre}. "
                f"Lluvia hoy: {lluvia_mm}mm, acumulado 7d: {rain_7d}mm, "
                f"prob lluvia intensa manana: {prob*100:.0f}%, riesgo: {nivel}/5 ({nivel_txt}). "
                f"Directo y util para ciudadanos. Sin markdown."
            )}]
        )
        return msg.content[0].text.strip()
    except Exception:
        return None


def alerta_reglas(nombre, lluvia_mm, prob, nivel, rain_7d):
    prob_pct = round(prob * 100)
    nivel_txt = {1:'muy bajo',2:'bajo',3:'medio',4:'alto',5:'muy alto'}.get(nivel,'medio')
    if nivel >= 5:
        return (f"ALERTA MAXIMA en {nombre}. "
                f"{'Se registran ' + str(lluvia_mm) + 'mm hoy con ' + str(rain_7d) + 'mm acumulados en 7 dias.' if lluvia_mm >= 10 else 'Acumulado semanal de ' + str(rain_7d) + 'mm y ' + str(prob_pct) + '% de probabilidad de lluvia intensa manana.'}"
                f" Riesgo MUY ALTO de inundacion.")
    elif nivel >= 4:
        return (f"Alerta en {nombre}. Lluvia de {lluvia_mm}mm hoy, acumulado {rain_7d}mm en 7 dias. "
                f"Riesgo {nivel_txt} de inundacion. Mantente informado.")
    elif nivel >= 3:
        return (f"Aviso preventivo en {nombre}. {lluvia_mm}mm registrados hoy. "
                f"Probabilidad de lluvia intensa manana: {prob_pct}%. Riesgo {nivel_txt}.")
    elif nivel >= 2:
        return f"{nombre}: Lluvia moderada ({lluvia_mm}mm). Riesgo {nivel_txt}. Sin alerta activa."
    else:
        return f"{nombre}: Condiciones normales. {lluvia_mm}mm hoy. Riesgo de inundacion muy bajo."


def generar_alerta(nombre, lluvia_mm, prob, nivel, rain_7d):
    return (alerta_watsonx(nombre, lluvia_mm, prob, nivel, rain_7d) or
            alerta_claude(nombre, lluvia_mm, prob, nivel, rain_7d) or
            alerta_reglas(nombre, lluvia_mm, prob, nivel, rain_7d))


# ── Ciclo de predicción ──────────────────────────────────
def run_ciclo():
    now = datetime.now(timezone.utc)
    hoy = now.date()
    print(f"[{now.strftime('%H:%M:%S UTC')}] Actualizando predicciones...")
    zones = {}

    for alcaldia_int, (lat, lon) in ALCALDIAS_COORDS.items():
        nombre_front = NAME_MAP[alcaldia_int]
        try:
            url = (f'https://api.open-meteo.com/v1/forecast'
                   f'?latitude={lat}&longitude={lon}'
                   f'&daily=precipitation_sum,temperature_2m_max,temperature_2m_min'
                   f'&timezone=America%2FMexico_City&past_days=30&forecast_days=1')
            d = requests.get(url, timeout=10).json()['daily']
            ll  = d['precipitation_sum']
            mx  = d['temperature_2m_max']
            mn  = d['temperature_2m_min']

            lluvia_mm = ll[-1] or 0.0
            maxt      = mx[-1] or 20.0
            mint      = mn[-1] or 12.0
            rain_3d   = sum(v or 0 for v in ll[-4:-1])
            rain_7d   = sum(v or 0 for v in ll[-8:-1])
            rain_30d  = sum(v or 0 for v in ll[:-1])
            mes       = hoy.month
            t_lluv    = 1 if 5 <= mes <= 10 else 0

            row = ref[ref['alcaldia'] == alcaldia_int]
            rp_max  = row.iloc[0]['riesgo_precip_max']  if not row.empty else 3
            rp_mean = row.iloc[0]['riesgo_precip_mean'] if not row.empty else 2.5

            # Agente 1
            enc1 = le_a1.transform([alcaldia_int])[0]
            X1 = pd.DataFrame([{'MONTH':mes,'DAY':hoy.day,'RAINFALL':lluvia_mm,
                                 'MAXT':maxt,'MINT':mint,'rain_3d':rain_3d,
                                 'rain_7d':rain_7d,'rain_30d':rain_30d,
                                 'temporada_lluvias':t_lluv,'riesgo_precip_max':rp_max,
                                 'riesgo_precip_mean':rp_mean,'alcaldia_encoded':enc1}])[FEAT_A1]
            prob_lluvia = round(float(modelo_a1.predict_proba(X1)[0][1]), 2)

            # Agente 2
            enc2 = le_a2.transform([alcaldia_int])[0]
            X2 = pd.DataFrame([{'alcaldia_encoded':enc2,'MONTH':mes,'DAY':hoy.day,
                                 'RAINFALL':lluvia_mm,'MAXT':maxt,'MINT':mint,
                                 'rain_3d':rain_3d,'rain_7d':rain_7d,
                                 'rain_30d':rain_30d,'temporada_lluvias':t_lluv}])[FEAT_A2]
            nivel_riesgo = max(1, min(5, round(float(modelo_a2.predict(X2)[0]))))

            # Agente 3
            alerta = generar_alerta(nombre_front, round(lluvia_mm,1),
                                    prob_lluvia, nivel_riesgo, round(rain_7d,1))

            zones[nombre_front] = {
                'lluvia_mm'   : round(lluvia_mm, 1),
                'prob_lluvia' : prob_lluvia,
                'nivel_riesgo': nivel_riesgo,
                'alerta'      : alerta,
            }
        except Exception as e:
            print(f"  ERROR {nombre_front}: {e}")
            zones[nombre_front] = {
                'lluvia_mm':0.0,'prob_lluvia':0.0,
                'nivel_riesgo':0,'alerta':'Datos no disponibles temporalmente.'
            }

    payload = {'timestamp': now.strftime('%Y-%m-%dT%H:%M:%SZ'), 'zones': zones}
    STATE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"  {len(zones)} zonas actualizadas OK")
    return payload


def loop_background():
    while True:
        try:
            run_ciclo()
        except Exception as e:
            print(f"ERROR loop: {e}")
        time.sleep(INTERVALO)


# ── FastAPI ──────────────────────────────────────────────
app = FastAPI(
    title="AquaInfer CDMX API",
    description="Sistema de predicción de riesgos de inundación urbana — Ciudad de México",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    # Primer ciclo inmediato, luego loop en background
    threading.Thread(target=loop_background, daemon=True).start()


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(BASE / "dashboard.html")


@app.get("/predict", summary="Predicciones en vivo para las 16 alcaldías")
def predict():
    """
    Devuelve el JSON con timestamp + zones en el formato del frontend.
    Se actualiza automáticamente cada 5 minutos.
    """
    if STATE.exists():
        return JSONResponse(json.loads(STATE.read_text(encoding='utf-8')))
    # Si no existe aún, generar en el momento
    return JSONResponse(run_ciclo())


@app.get("/predict/{zona}", summary="Predicción para una zona específica")
def predict_zona(zona: str):
    """Devuelve predicción solo para la zona indicada."""
    if STATE.exists():
        data = json.loads(STATE.read_text(encoding='utf-8'))
        if zona in data['zones']:
            return JSONResponse({
                'timestamp': data['timestamp'],
                'zona'     : zona,
                **data['zones'][zona]
            })
    return JSONResponse({'error': f'Zona "{zona}" no encontrada'}, status_code=404)


@app.get("/health", summary="Health check")
def health():
    last = None
    if STATE.exists():
        data = json.loads(STATE.read_text(encoding='utf-8'))
        last = data.get('timestamp')
    return {'status': 'ok', 'last_update': last, 'version': '1.0.0'}
