# Deploy su Render - Guida Completa 🚀

## Prerequisiti

1. Account **Render** gratuito: [render.com](https://render.com)
2. Repository GitHub con questo progetto
3. API Keys pronte:
   - `API_FOOTBALL_KEY` da [api-football.com](https://api-football.com)
   - `FOOTBALL_DATA_TOKEN` da [football-data.org](https://football-data.org)

## Step 1️⃣: Push su GitHub

```bash
git init
git add .
git commit -m "Initial commit - Football Betting API"
git branch -M main
git remote add origin https://github.com/TUO-USERNAME/TUO-REPO.git
git push -u origin main
```

## Step 2️⃣: Deploy su Render

1. **Vai su [Render Dashboard](https://dashboard.render.com)**

2. **Clicca su "New +" → "Web Service"**

3. **Connetti GitHub repository**
   - Autorizza Render ad accedere ai tuoi repo
   - Seleziona il repository del progetto

4. **Configurazione automatica**
   - Render rileverà automaticamente `render.yaml`
   - Nome: `football-betting-api` (o quello che preferisci)
   - Environment: `Python 3`
   - Build Command: `./build.sh`
   - Start Command: `gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT`

5. **Aggiungi Environment Variables** (Sezione "Environment")
   
   **OBBLIGATORIE:**
   ```
   API_FOOTBALL_KEY=tua_chiave_qui
   FOOTBALL_DATA_TOKEN=tua_chiave_qui
   ```
   
   **OPZIONALI (già con default):**
   ```
   LEAGUE_IDS=39,140,135,78,61
   POISSON_WEIGHT=0.4
   XGBOOST_WEIGHT=0.6
   MIN_EDGE=0.03
   KELLY_FRACTION=0.25
   ```

6. **Seleziona piano FREE**
   - 750 ore/mese gratis
   - Sleep dopo 15 min inattività
   - Sufficiente per progetti personali

7. **Clicca "Create Web Service"**

## Step 3️⃣: Primo Deploy

Render farà automaticamente:
- ✅ Build dell'immagine Python
- ✅ Installazione dipendenze (`requirements.txt`)
- ✅ Avvio server Gunicorn
- ✅ Health check su `/health`

**Tempo stimato:** 5-10 minuti

## Step 4️⃣: Trainare i modelli

Una volta che il servizio è **Live**, devi trainare i modelli:

**URL del tuo servizio:** `https://football-betting-api-xxx.onrender.com`

### Opzione A: Tramite Browser
1. Apri `https://TUO-URL.onrender.com/docs`
2. Cerca endpoint `POST /api/train-current`
3. Clicca "Try it out" → "Execute"
4. Aspetta ~2-3 minuti (scarica dati + training)

### Opzione B: Tramite cURL
```bash
curl -X POST "https://TUO-URL.onrender.com/api/train-current"
```

**Response di successo:**
```json
{
  "status": "success",
  "message": "Models trained with 3 seasons: [2025, 2024, 2023]",
  "details": {
    "matches_used": 4259,
    "leagues": [39, 140, 135, 78, 61],
    "models": {
      "poisson": {"fitted": true, "weight": 0.4},
      "xgboost": {"fitted": true, "weight": 0.6},
      "ensemble_ready": true
    }
  }
}
```

⚠️ **IMPORTANTE:** I modelli non persistono tra restart. Ogni volta che Render riavvia il servizio (dopo sleep o deploy), devi ri-trainare.

## Step 5️⃣: Testare le API

### Health Check
```bash
curl https://TUO-URL.onrender.com/health
```

### Partite Upcoming (Nuova!)
```bash
# Tutte le leghe configurate, prossimi 7 giorni
curl "https://TUO-URL.onrender.com/api/upcoming-matches"

# Solo Premier League, prossimi 3 giorni
curl "https://TUO-URL.onrender.com/api/upcoming-matches?league_ids=39&days_ahead=3"
```

**Response:**
```json
{
  "count": 45,
  "leagues": [
    {"id": 39, "name": "Premier League"},
    {"id": 140, "name": "La Liga"}
  ],
  "days_ahead": 7,
  "matches": [
    {
      "fixture_id": 12345,
      "date": "2025-12-21T15:00:00",
      "league": {"id": 39, "name": "Premier League"},
      "home_team": {"id": 33, "name": "Manchester United"},
      "away_team": {"id": 34, "name": "Newcastle"},
      "status": "NS"
    }
  ]
}
```

### Predizione Singola Partita
```bash
curl -X POST "https://TUO-URL.onrender.com/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team_id": 33,
    "away_team_id": 34,
    "home_team_name": "Manchester United",
    "away_team_name": "Newcastle",
    "league_id": 39
  }'
```

### Value Bet Analysis
```bash
curl -X POST "https://TUO-URL.onrender.com/api/value-bet" \
  -H "Content-Type: application/json" \
  -d '{
    "home_team_id": 33,
    "away_team_id": 34,
    "home_team_name": "Manchester United",
    "away_team_name": "Newcastle",
    "league_id": 39,
    "odds": {"1": 2.10, "X": 3.40, "2": 3.50}
  }'
```

## Frontend Integration 🌐

### 1. Deploy Frontend (Vercel/Netlify - GRATIS)

**Vercel:**
- Vai su [vercel.com](https://vercel.com)
- Importa repo GitHub del frontend
- Deploy automatico

**Netlify:**
- Vai su [netlify.com](https://netlify.com)
- Drop & drag della cartella build
- Deploy in 1 minuto

### 2. Chiamate dal Frontend (JavaScript)

```javascript
// Backend URL
const API_URL = "https://football-betting-api-xxx.onrender.com";

// 1. Recupera partite upcoming
async function getUpcomingMatches(leagueIds = "39,140,135", daysAhead = 7) {
  const response = await fetch(
    `${API_URL}/api/upcoming-matches?league_ids=${leagueIds}&days_ahead=${daysAhead}`
  );
  const data = await response.json();
  return data.matches; // Array di partite
}

// 2. Predizione per una partita
async function predictMatch(match) {
  const response = await fetch(`${API_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      home_team_id: match.home_team.id,
      away_team_id: match.away_team.id,
      home_team_name: match.home_team.name,
      away_team_name: match.away_team.name,
      league_id: match.league.id
    })
  });
  return await response.json();
}

// 3. Workflow completo
async function analyzeAllMatches() {
  // Recupera partite
  const matches = await getUpcomingMatches();
  
  // Predici tutte
  const predictions = [];
  for (const match of matches) {
    const prediction = await predictMatch(match);
    predictions.push({
      match: match,
      prediction: prediction
    });
  }
  
  // Ordina per confidence/probabilità
  predictions.sort((a, b) => 
    parseFloat(b.prediction.probabilities.home_win) - 
    parseFloat(a.prediction.probabilities.home_win)
  );
  
  return predictions;
}
```

### 3. CORS già configurato ✅

Il tuo backend ha già CORS abilitato in `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accetta da qualsiasi dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Limiti Piano FREE Render

- ✅ 750 ore/mese (sufficiente per 24/7)
- ⚠️ Sleep dopo 15 min inattività (primo request ~10-15 sec wake-up)
- ⚠️ 512 MB RAM (sufficiente per questo progetto)
- ⚠️ Modelli non persistono tra restart → ri-trainare dopo ogni deploy

## Soluzioni ai Limiti

### 1. Evitare Sleep
**Opzione A:** Cron job che pinga ogni 10 min
```bash
# Cron gratuito: cron-job.org
*/10 * * * * curl https://TUO-URL.onrender.com/health
```

**Opzione B:** UptimeRobot gratuito (monitoraggio ogni 5 min)

### 2. Persistenza Modelli
**Opzione A:** Salvare modelli su cloud storage (S3, Google Cloud Storage)
**Opzione B:** Re-train automatico al startup (aggiungi al `lifespan` in `main.py`)

## Monitoring

### Dashboard Render
- Logs in tempo reale
- Metriche CPU/RAM
- Deploy history

### Endpoint Health
```bash
curl https://TUO-URL.onrender.com/health
```

Response:
```json
{
  "status": "healthy",
  "models": {
    "poisson": {"fitted": true},
    "xgboost": {"fitted": true},
    "ensemble_ready": true
  },
  "api_configured": true,
  "timestamp": "2025-12-20T..."
}
```

## Troubleshooting

### Deploy fallisce
1. Controlla logs su Render Dashboard
2. Verifica `requirements.txt` completo
3. Check Python version (3.11 in `render.yaml`)

### Modelli non fitted
1. Chiama `POST /api/train-current` dopo deploy
2. Check environment variables (API keys)

### Sleep troppo frequente
1. Usa UptimeRobot per ping automatico
2. Oppure upgrade a piano paid ($7/mese - no sleep)

## Costi Stimati

- **Backend Render Free:** $0/mese
- **Frontend Vercel/Netlify:** $0/mese
- **API-Football:** $0/mese (100 req/giorno)
- **Football-data.org:** $0/mese (10 req/minuto)

**TOTALE: GRATIS! 🎉**

## Next Steps

1. ✅ Deploy backend su Render
2. ✅ Train modelli via `/api/train-current`
3. ✅ Test endpoints con Postman/cURL
4. 🚀 Crea frontend React/Vue/Vanilla JS
5. 🚀 Deploy frontend su Vercel/Netlify
6. 🚀 Integra le chiamate API

**Domande?** Check la documentazione interattiva: `https://TUO-URL.onrender.com/docs`
