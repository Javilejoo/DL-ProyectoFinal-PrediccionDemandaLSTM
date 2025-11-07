import numpy as np
import pandas as pd
from datetime import date, timedelta
from calendar import monthrange

# ===============================
# CONFIG
# ===============================
SEED = 42
np.random.seed(SEED)

START = date(2018, 1, 1)      # lunes
END   = date(2024, 12, 30)     # lunes

# Nivel y tendencia
BASE_LEVEL = 45.0
ANNUAL_TREND = 0.03            # +3% anual, multiplicativa

# Estacionalidad mensual (ajústala a tu producto)
MONTH_SEASON = {
    1:  -0.05,
    2:   0.10,  # San Valentín
    3:   0.00,
    4:   0.02,
    5:   0.06,  # Día de la Madre
    6:  -0.02,
    7:  -0.03,
    8:   0.00,
    9:   0.01,
    10:  0.03,
    11:  0.05,  # Black Friday
    12:  0.14,  # fin de año
}

# Estacionalidad semanal (Fourier, suave)
SEASON_WOY_WEIGHT = 0.05       # peso de la componente semanal

# Promos
PROMO_BASE_PROB = 0.12         # prob. base de promo
PROMO_Q4_BOOST  = 0.08         # más probabilidad en Q4 (oct-dic)
PROMO_LIFT_RANGE = (0.10, 0.40)    # +10%..+40% directo
PROMO_CARRYOVER  = (0.20, 0.40)    # 20%..40% del lift en la semana siguiente

# Eventos (lifts puntuales, se aplican a la semana "core")
EVENT_VALENTINE  = (0.40, 1.20)  # 14/feb
EVENT_MOTHERSDAY = (0.30, 0.80)  # 10/may (Guatemala)
EVENT_CHRISTMAS  = (0.20, 0.70)  # 25/dic
EVENT_NEWYEAR    = (0.10, 0.40)  # 1/ene (del año siguiente si cae en la semana)
EVENT_BLACKFRI   = (0.20, 0.60)  # último viernes de noviembre

# Lead/Decay de eventos
LEAD_RANGE  = (0.15, 0.35)    # 15–35% del core en la semana previa
DECAY_RANGE = (0.10, 0.25)    # 10–25% del core en la semana posterior (opcional)

# Ruido / conteo
NOISE_SD = 0.10               # lognormal multiplicativo
NB_ALPHA = 15.0               # dispersión de Negative Binomial (↑ => menos dispersión)
DEMAND_FLOOR = 0              # mínimo
DEMAND_CAP   = None           # tope (None o int)

# ===============================
# HELPERS
# ===============================
def weeks_mondays(start: date, end: date):
    d = start
    weeks = []
    while d <= end:
        weeks.append(d)
        d += timedelta(days=7)
    return weeks

def week_contains(day: date, week_start: date) -> bool:
    return week_start <= day <= (week_start + timedelta(days=6))

def last_friday_of_november(y: int) -> date:
    last_day = monthrange(y, 11)[1]
    d = date(y, 11, last_day)
    while d.weekday() != 4:  # 4 = viernes
        d -= timedelta(days=1)
    return d

def event_lift_for_week(week_start: date) -> float:
    """Suma lifts de eventos que caen en la semana (núcleo/core)."""
    y = week_start.year
    lift = 0.0
    # San Valentín
    if week_contains(date(y, 2, 14), week_start):
        lift += np.random.uniform(*EVENT_VALENTINE)
    # Día de la Madre (Guatemala 10/mayo)
    if week_contains(date(y, 5, 10), week_start):
        lift += np.random.uniform(*EVENT_MOTHERSDAY)
    # Navidad
    if week_contains(date(y, 12, 25), week_start):
        lift += np.random.uniform(*EVENT_CHRISTMAS)
    # Año Nuevo (1/ene del año siguiente)
    if week_contains(date(y + 1, 1, 1), week_start):
        lift += np.random.uniform(*EVENT_NEWYEAR)
    # Black Friday
    if week_contains(last_friday_of_november(y), week_start):
        lift += np.random.uniform(*EVENT_BLACKFRI)
    return lift

def negbinomial_from_mu_alpha(mu, alpha):
    """
    Devuelve un entero ~ Negative Binomial con media mu y parámetro alpha (dispersion).
    var = mu + mu^2/alpha; p = alpha/(alpha+mu); r = alpha
    """
    mu = max(1e-6, mu)
    p = alpha / (alpha + mu)
    r = alpha
    return np.random.negative_binomial(r, p)

# ===============================
# DATAFRAME BASE
# ===============================
weeks = weeks_mondays(START, END)
df = pd.DataFrame({"week_start": weeks})
dt = pd.to_datetime(df["week_start"])
df["year"] = dt.dt.year
df["month"] = dt.dt.month
df["week_of_year"] = dt.dt.isocalendar().week.astype(int)

# Tendencia anual multiplicativa
years_from_start = df["year"] - df["year"].min()
trend = (1 + ANNUAL_TREND) ** years_from_start

# Estacionalidad mensual
season_month = df["month"].map(MONTH_SEASON).astype(float)

# Estacionalidad semanal con Fourier (1 armónico)
woy = df["week_of_year"].values
sin_woy = np.sin(2*np.pi*woy/52.0)
cos_woy = np.cos(2*np.pi*woy/52.0)
season_woy = SEASON_WOY_WEIGHT * (0.7 * sin_woy + 0.3 * cos_woy)

# Eventos core (semana del feriado)
df["event_core"] = df["week_start"].apply(event_lift_for_week)
df["holiday_flag"] = (df["event_core"] > 0).astype(int)  # semana del evento

# Lead & Decay
event_core = df["event_core"].values.astype(float)
lead = np.zeros_like(event_core)
decay = np.zeros_like(event_core)
for i, core in enumerate(event_core):
    if core > 0:
        # semana previa (lead)
        if i - 1 >= 0:
            lead_factor = np.random.uniform(*LEAD_RANGE)
            lead[i - 1] += core * lead_factor
        # semana siguiente (decay)
        if i + 1 < len(event_core):
            decay_factor = np.random.uniform(*DECAY_RANGE)
            decay[i + 1] += core * decay_factor
df["event_lead"]  = lead
df["event_decay"] = decay
df["event_lift_total"] = df["event_core"] + df["event_lead"] + df["event_decay"]
df["holiday_lead_flag"]  = (df["event_lead"]  > 0).astype(int)  # semana previa
df["holiday_decay_flag"] = (df["event_decay"] > 0).astype(int)  # semana posterior

# Promos con probabilidad dependiente de temporada (Q4 más probable)
q4 = df["month"].isin([10, 11, 12]).astype(float)
promo_prob = PROMO_BASE_PROB + q4 * PROMO_Q4_BOOST
promo_mask = np.random.rand(len(df)) < promo_prob
df["promo_flag"] = promo_mask.astype(int)

# Intensidad de promo y carryover
promo_lift = np.zeros(len(df))
promo_lift[promo_mask] = np.random.uniform(*PROMO_LIFT_RANGE, size=promo_mask.sum())
df["promo_lift"] = promo_lift

carryover = np.zeros(len(df))
for i in range(len(df) - 1):
    if promo_mask[i]:
        carry = np.random.uniform(*PROMO_CARRYOVER)
        carryover[i + 1] += promo_lift[i] * carry
df["promo_carryover"] = carryover

# Media "latente" multiplicativa
mu_base = (
    BASE_LEVEL
    * (1 + season_month.values)
    * (1 + season_woy)
    * trend.values
    * (1 + df["event_lift_total"].values)     # core + lead + decay
    * (1 + df["promo_lift"].values + df["promo_carryover"].values)
)

# Ruido multiplicativo lognormal y media observada
noise = np.random.lognormal(mean=0.0, sigma=NOISE_SD, size=len(df))
mu_obs = np.clip(mu_base * noise, a_min=1.0, a_max=None)

# Conteo entero con sobre-dispersión (Negative Binomial)
demand = [negbinomial_from_mu_alpha(m, NB_ALPHA) for m in mu_obs]
if DEMAND_CAP is not None:
    demand = [min(max(DEMAND_FLOOR, x), DEMAND_CAP) for x in demand]
else:
    demand = [max(DEMAND_FLOOR, x) for x in demand]
df["demand"] = np.array(demand, dtype=int)

# ===============================
# SALIDAS
# ===============================

# 1) Dataset 
out = df[[
    "week_start", "demand",
    "promo_flag", "promo_lift", "promo_carryover",
    "holiday_flag", "holiday_lead_flag", "holiday_decay_flag",
    "year", "month", "week_of_year"
]]

out.to_csv("demand_weekly_chocolates_2018-2024.csv", index=False)