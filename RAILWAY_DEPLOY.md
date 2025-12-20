# Deploy su Railway - Guida Rapida 🚀

## 🎁 Piano FREE Railway (Migliore di Render!)

- ✅ **$5 credito GRATIS al mese** (circa 500 ore)
- ✅ **NO sleep automatico** (sempre attivo!)
- ✅ **512 MB RAM** inclusi
- ✅ **Deploy illimitati**
- ✅ **SSL automatico**
- ✅ **Deployment più veloce** (~2-3 min)

---

## 📋 Step-by-Step

### 1️⃣ Crea Account Railway

1. Vai su **[railway.app](https://railway.app)**
2. Clicca **"Start a New Project"**
3. Login con **GitHub** (consigliato)

### 2️⃣ Deploy da GitHub

1. **Dashboard Railway** → **"New Project"**
2. Seleziona **"Deploy from GitHub repo"**
3. Autorizza Railway ad accedere ai tuoi repository
4. Seleziona il repository **`betting`**
5. Railway rileva automaticamente Python e configura tutto! ✨

### 3️⃣ Aggiungi Environment Variables

Nella dashboard del progetto:

1. Vai su **"Variables"** tab
2. Clicca **"+ New Variable"**
3. Aggiungi queste variabili:

```bash
# OBBLIGATORIE
API_FOOTBALL_KEY=tua_chiave_da_api_football_com
FOOTBALL_DATA_TOKEN=tua_chiave_da_football_data_org

# OPZIONALI (con default già configurati)
LEAGUE_IDS=39,140,135,78,61
POISSON_WEIGHT=0.4
XGBOOST_WEIGHT=0.6
MIN_EDGE=0.03
KELLY_FRACTION=0.25
FORM_MATCHES=5
CACHE_TTL=3600
```

4. Clicca **"Deploy"** (o aspetta auto-deploy)

### 4️⃣ Ottieni URL Pubblico

1. Nella dashboard, vai su **"Settings"** → **"Networking"**
2. Clicca **"Generate Domain"**
3. Riceverai un URL tipo: `https://betting-production-xxx.up.railway.app`

---

## 🏋️ Train dei Modelli (IMPORTANTE!)

Una volta deployato, **DEVI trainare i modelli**:

### Opzione A: Tramite Browser (più facile)
1. Apri: `https://TUO-URL.up.railway.app/docs`
2. Cerca **`POST /api/train-current`**
3. Clicca **"Try it out"** → **"Execute"**
4. Aspetta 2-3 minuti

### Opzione B: Tramite cURL
```bash
curl -X POST "https://TUO-URL.up.railway.app/api/train-current"
```

**Response di successo:**
```json
{
  "status": "success",
  "message": "Models trained with 3 seasons: [2025, 2024, 2023]",
  "details": {
    "matches_used": 4259,
    "models": {
      "ensemble_ready": true
    }
  }
}
```

---

## 🧪 Test delle API

### Health Check
```bash
curl https://TUO-URL.up.railway.app/health
```

### Upcoming Matches (NUOVO!)
```bash
# Tutte le league, prossimi 7 giorni
curl "https://TUO-URL.up.railway.app/api/upcoming-matches"

# Solo Premier League + La Liga, prossimi 3 giorni
curl "https://TUO-URL.up.railway.app/api/upcoming-matches?league_ids=39,140&days_ahead=3"
```

**Response:**
```json
{
  "count": 45,
  "leagues": [
    {"id": 39, "name": "Premier League"},
    {"id": 140, "name": "La Liga"}
  ],
  "matches": [
    {
      "fixture_id": 12345,
      "date": "2025-12-21T15:00:00",
      "home_team": {"id": 33, "name": "Manchester United"},
      "away_team": {"id": 34, "name": "Newcastle"},
      "league": {"id": 39, "name": "Premier League"}
    }
  ]
}
```

### Predizione Singola
```bash
curl -X POST "https://TUO-URL.up.railway.app/api/predict" \
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
curl -X POST "https://TUO-URL.up.railway.app/api/value-bet" \
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

---

## 🌐 Integrazione Frontend

### Deploy Frontend GRATIS su:
- **Vercel** ([vercel.com](https://vercel.com)) ← Consigliato per React/Next.js
- **Netlify** ([netlify.com](https://netlify.com)) ← Consigliato per siti statici
- **GitHub Pages** ← Gratis ma solo siti statici

### Esempio JavaScript Vanilla

```html
<!DOCTYPE html>
<html>
<head>
    <title>Football Predictions</title>
    <style>
        body { font-family: Arial; max-width: 1200px; margin: 20px auto; }
        .match { border: 1px solid #ddd; padding: 15px; margin: 10px 0; }
        .prediction { background: #f0f0f0; padding: 10px; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>⚽ Football Predictions</h1>
    <button onclick="loadMatches()">Carica Partite Upcoming</button>
    <div id="matches"></div>

    <script>
        const API_URL = "https://TUO-URL.up.railway.app";

        async function loadMatches() {
            const container = document.getElementById('matches');
            container.innerHTML = '<p>Caricamento...</p>';

            // 1. Recupera partite upcoming
            const response = await fetch(`${API_URL}/api/upcoming-matches?days_ahead=7`);
            const data = await response.json();

            container.innerHTML = '';

            // 2. Per ogni partita, richiedi predizione
            for (const match of data.matches.slice(0, 10)) { // Prime 10
                const prediction = await predictMatch(match);
                displayMatch(match, prediction);
            }
        }

        async function predictMatch(match) {
            const response = await fetch(`${API_URL}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
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

        function displayMatch(match, prediction) {
            const container = document.getElementById('matches');
            const div = document.createElement('div');
            div.className = 'match';
            
            const probs = prediction.probabilities;
            const date = new Date(match.date).toLocaleString('it-IT');

            div.innerHTML = `
                <h3>${match.home_team.name} vs ${match.away_team.name}</h3>
                <p><strong>League:</strong> ${match.league.name}</p>
                <p><strong>Data:</strong> ${date}</p>
                <div class="prediction">
                    <h4>Predizione: ${prediction.prediction}</h4>
                    <p>🏠 Home Win: ${probs.home_win}%</p>
                    <p>🤝 Draw: ${probs.draw}%</p>
                    <p>✈️ Away Win: ${probs.away_win}%</p>
                    <p><strong>Confidence:</strong> ${prediction.confidence}</p>
                </div>
            `;
            container.appendChild(div);
        }
    </script>
</body>
</html>
```

### Esempio React

```jsx
import { useState, useEffect } from 'react';

const API_URL = "https://TUO-URL.up.railway.app";

function App() {
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadMatches = async () => {
    setLoading(true);
    
    // 1. Fetch upcoming matches
    const res = await fetch(`${API_URL}/api/upcoming-matches?days_ahead=7`);
    const data = await res.json();
    
    // 2. Get predictions for each match
    const predictions = await Promise.all(
      data.matches.slice(0, 10).map(async (match) => {
        const pred = await fetch(`${API_URL}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            home_team_id: match.home_team.id,
            away_team_id: match.away_team.id,
            home_team_name: match.home_team.name,
            away_team_name: match.away_team.name,
            league_id: match.league.id
          })
        });
        return { match, prediction: await pred.json() };
      })
    );
    
    setMatches(predictions);
    setLoading(false);
  };

  return (
    <div className="App">
      <h1>⚽ Football Predictions</h1>
      <button onClick={loadMatches}>Carica Partite</button>
      
      {loading && <p>Caricamento...</p>}
      
      {matches.map(({ match, prediction }) => (
        <div key={match.fixture_id} className="match-card">
          <h3>{match.home_team.name} vs {match.away_team.name}</h3>
          <p>{match.league.name}</p>
          <div className="prediction">
            <p>Home: {prediction.probabilities.home_win}%</p>
            <p>Draw: {prediction.probabilities.draw}%</p>
            <p>Away: {prediction.probabilities.away_win}%</p>
            <p>Prediction: {prediction.prediction}</p>
          </div>
        </div>
      ))}
    </div>
  );
}

export default App;
```

---

## 📊 Monitoring Railway

### Dashboard Railway:
- **Logs in tempo reale** (tab "Deployments")
- **Metriche CPU/RAM** (tab "Metrics")
- **Deploy history**
- **Environment variables**

### Health Check:
```bash
# Check se tutto funziona
curl https://TUO-URL.up.railway.app/health
```

Response OK:
```json
{
  "status": "healthy",
  "models": {
    "poisson": {"fitted": true},
    "xgboost": {"fitted": true},
    "ensemble_ready": true
  },
  "api_configured": true
}
```

---

## ⚠️ Note Importanti

### Persistenza Modelli
I modelli **NON persistono** tra restart. Ogni volta che Railway rideploya:
1. Chiama `POST /api/train-current` per ri-trainare
2. Oppure implementa auto-train al startup (modifica `main.py`)

### Limiti Piano FREE
- ✅ $5 credito/mese (~500 ore)
- ✅ NO sleep automatico
- ⚠️ Se esaurisci credito → servizio si ferma fino al mese dopo
- 💡 Monitora l'uso nella dashboard

### Ottimizzazione Costi
- Usa cache TTL alto (3600s = 1h)
- Evita chiamate API inutili
- Considera upgrade a $5/mese per credito illimitato

---

## 🆚 Railway vs Render

| Feature | Railway FREE | Render FREE |
|---------|-------------|-------------|
| **Credito** | $5/mese (~500h) | 750h/mese |
| **Sleep** | ❌ Mai | ✅ Dopo 15 min |
| **RAM** | 512 MB | 512 MB |
| **Deploy** | Veloce (2-3 min) | Lento (5-10 min) |
| **Persistenza** | ✅ Sempre attivo | ⚠️ Wake-up 15s |
| **Migliore per** | 🏆 Progetti seri | Testing/Hobby |

**Railway vince!** 🎉

---

## 🚀 Quick Start Completo

```bash
# 1. Push su GitHub (se non fatto)
git add .
git commit -m "Add Railway config"
git push

# 2. Vai su railway.app
# 3. New Project → Deploy from GitHub
# 4. Seleziona repo "betting"
# 5. Aggiungi env variables (API_FOOTBALL_KEY, FOOTBALL_DATA_TOKEN)
# 6. Aspetta deploy (2-3 min)
# 7. Generate Domain (Settings → Networking)
# 8. Train modelli: curl -X POST https://TUO-URL.up.railway.app/api/train-current
# 9. Test: curl https://TUO-URL.up.railway.app/api/upcoming-matches
```

**TOTALE TEMPO: ~10 minuti! ⚡**

---

## 💰 Costi Finali

- **Backend Railway:** $0/mese (piano FREE)
- **Frontend Vercel/Netlify:** $0/mese
- **API-Football:** $0/mese (100 req/giorno)
- **Football-data.org:** $0/mese

**TUTTO GRATIS! 🎊**

---

## 🆘 Troubleshooting

### Deploy fallisce
- Check logs in tab "Deployments"
- Verifica `requirements.txt` completo
- Check Python version in `nixpacks.toml`

### Modelli non fitted
- Chiama `/api/train-current` dopo ogni deploy
- Verifica API keys in "Variables"

### Credito esaurito
- Monitora in Dashboard → Usage
- Considera upgrade a $5/mese

---

## 📚 Documentazione API Completa

Una volta deployato, vai su:
```
https://TUO-URL.up.railway.app/docs
```

Troverai l'interfaccia **Swagger interattiva** con tutti gli endpoint! 🎯
