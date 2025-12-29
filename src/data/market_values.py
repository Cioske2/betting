"""
Market Values Static Data Module.

Contains squad market values (in Euros) for major European leagues.
Data source: Transfermarkt estimated values for 2025/2026 season.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Market values in millions (EUR)
# All values will be converted to full Euros in the lookup function
MARKET_VALUES = {
    # Premier League
    "Arsenal": 1290.0,
    "Manchester City": 1190.0,
    "Chelsea": 1180.0,
    "Liverpool": 1040.0,
    "Tottenham Hotspur": 878.5,
    "Manchester United": 719.1,
    "Newcastle United": 712.9,
    "Nottingham Forest": 619.6,
    "Aston Villa": 519.5,
    "Brighton & Hove Albion": 519.5,
    "Crystal Palace": 501.6,
    "AFC Bournemouth": 483.8,
    "Brentford": 455.9,
    "West Ham United": 420.5,
    "Everton": 340.2,
    "Fulham": 331.6,
    "Wolverhampton Wanderers": 315.8,
    "Leeds United": 285.5,
    "Sunderland": 248.0,
    "Burnley": 210.0,
    
    # La Liga
    "Real Madrid": 1380.0,
    "FC Barcelona": 1120.0,
    "Atlético de Madrid": 589.0,
    "Real Sociedad": 385.4,
    "Athletic Bilbao": 303.0,
    "Villarreal": 260.0,
    "Girona FC": 245.5,
    "Valencia CF": 225.0,
    "Real Betis": 210.0,
    "Sevilla FC": 195.8,
    "Celta de Vigo": 165.2,
    "CA Osasuna": 140.5,
    "RCD Mallorca": 125.0,
    "Getafe CF": 115.6,
    "Deportivo Alavés": 105.0,
    "RCD Espanyol": 98.4,
    "Rayo Vallecano": 92.5,
    "Elche CF": 85.0,
    "Real Oviedo": 75.0,
    "Levante UD": 70.0,
    
    # Serie A
    "Inter Milan": 670.3,
    "Juventus": 551.7,
    "AC Milan": 485.5,
    "SSC Napoli": 466.8,
    "AS Roma": 320.0,
    "Atalanta": 315.0,
    "SS Lazio": 245.0,
    "Fiorentina": 235.0,
    "Bologna": 210.0,
    "Torino": 185.5,
    "Monza": 145.0,
    "Como 1907": 223.0,
    "Genoa": 135.8,
    "Udinese": 128.4,
    "Parma": 120.0,
    "Cagliari": 110.5,
    "Lecce": 95.0,
    "Hellas Verona": 88.6,
    "Sassuolo": 115.0,
    "Pisa": 65.0,
    
    # Bundesliga
    "Bayern Munich": 980.6,
    "Borussia Dortmund": 511.4,
    "Bayer Leverkusen": 445.4,
    "RB Leipzig": 435.0,
    "Eintracht Frankfurt": 285.0,
    "VfB Stuttgart": 240.0,
    "Wolfsburg": 220.5,
    "Borussia M'gladbach": 195.0,
    "Freiburg": 188.4,
    "Hoffenheim": 165.0,
    "Union Berlin": 155.8,
    "Werder Bremen": 145.0,
    "Mainz 05": 120.4,
    "Augsburg": 115.0,
    "FC Köln": 105.0,
    "Hamburger SV": 95.0,
    "Heidenheim": 82.5,
    "St. Pauli": 65.0,
    
    # Ligue 1
    "Paris Saint-Germain": 1190.0,
    "Olympique Marseille": 397.8,
    "AS Monaco": 339.5,
    "RC Strasbourg": 317.4,
    "LOSC Lille": 295.0,
    "Olympique Lyon": 265.0,
    "Stade Rennais": 240.0,
    "OGC Nice": 215.5,
    "RC Lens": 198.0,
    "Stade Reims": 165.0,
    "Toulouse FC": 145.4,
    "Montpellier HSC": 115.0,
    "Stade Brestois": 110.0,
    "FC Lorient": 98.5,
    "FC Nantes": 92.0,
    "Le Havre AC": 75.0,
    "Angers SCO": 65.0,
    "AJ Auxerre": 58.0,
}

# Team name aliases - maps common variations to canonical names
TEAM_ALIASES = {
    # Premier League aliases
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Spurs": "Tottenham Hotspur",
    "Tottenham": "Tottenham Hotspur",
    "Newcastle": "Newcastle United",
    "Brighton": "Brighton & Hove Albion",
    "Bournemouth": "AFC Bournemouth",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Nottm Forest": "Nottingham Forest",
    "Forest": "Nottingham Forest",
    "Leeds": "Leeds United",
    
    # La Liga aliases
    "Barcelona": "FC Barcelona",
    "Barca": "FC Barcelona",
    "Atletico Madrid": "Atlético de Madrid",
    "Atletico": "Atlético de Madrid",
    "Athletic Club": "Athletic Bilbao",
    "Bilbao": "Athletic Bilbao",
    "Sociedad": "Real Sociedad",
    "Betis": "Real Betis",
    "Sevilla": "Sevilla FC",
    "Valencia": "Valencia CF",
    "Celta Vigo": "Celta de Vigo",
    "Celta": "Celta de Vigo",
    "Osasuna": "CA Osasuna",
    "Mallorca": "RCD Mallorca",
    "Getafe": "Getafe CF",
    "Alaves": "Deportivo Alavés",
    "Espanyol": "RCD Espanyol",
    "Girona": "Girona FC",
    "Elche": "Elche CF",
    "Rayo": "Rayo Vallecano",
    "Levante": "Levante UD",
    
    # Serie A aliases
    "Inter": "Inter Milan",
    "Internazionale": "Inter Milan",
    "FC Internazionale": "Inter Milan",
    "Milan": "AC Milan",
    "Juve": "Juventus",
    "Napoli": "SSC Napoli",
    "Roma": "AS Roma",
    "Lazio": "SS Lazio",
    "Verona": "Hellas Verona",
    "Como": "Como 1907",
    
    # Bundesliga aliases
    "Bayern": "Bayern Munich",
    "FC Bayern": "Bayern Munich",
    "Dortmund": "Borussia Dortmund",
    "BVB": "Borussia Dortmund",
    "Leverkusen": "Bayer Leverkusen",
    "Bayer 04": "Bayer Leverkusen",
    "Leipzig": "RB Leipzig",
    "Frankfurt": "Eintracht Frankfurt",
    "Stuttgart": "VfB Stuttgart",
    "Gladbach": "Borussia M'gladbach",
    "Monchengladbach": "Borussia M'gladbach",
    "Mönchengladbach": "Borussia M'gladbach",
    "Bremen": "Werder Bremen",
    "Mainz": "Mainz 05",
    "Koln": "FC Köln",
    "Cologne": "FC Köln",
    "Köln": "FC Köln",
    "Hamburg": "Hamburger SV",
    "HSV": "Hamburger SV",
    
    # Ligue 1 aliases
    "PSG": "Paris Saint-Germain",
    "Paris SG": "Paris Saint-Germain",
    "Paris": "Paris Saint-Germain",
    "Marseille": "Olympique Marseille",
    "OM": "Olympique Marseille",
    "Lyon": "Olympique Lyon",
    "OL": "Olympique Lyon",
    "Monaco": "AS Monaco",
    "Lille": "LOSC Lille",
    "Nice": "OGC Nice",
    "Lens": "RC Lens",
    "Strasbourg": "RC Strasbourg",
    "Rennes": "Stade Rennais",
    "Reims": "Stade Reims",
    "Toulouse": "Toulouse FC",
    "Montpellier": "Montpellier HSC",
    "Brest": "Stade Brestois",
    "Lorient": "FC Lorient",
    "Nantes": "FC Nantes",
    "Le Havre": "Le Havre AC",
    "Angers": "Angers SCO",
    "Auxerre": "AJ Auxerre",
}

# Default value for unknown teams (50 million EUR)
DEFAULT_VALUE_MILLIONS = 50.0


def get_team_value(team_name: str) -> float:
    """
    Get the market value of a team in Euros.
    
    Args:
        team_name: The team name (handles aliases automatically)
        
    Returns:
        Market value in Euros (not millions). Returns 50M default for unknown teams.
    """
    if not team_name:
        return DEFAULT_VALUE_MILLIONS * 1_000_000
    
    # Clean the team name
    clean_name = team_name.strip()
    
    # Try direct lookup first
    if clean_name in MARKET_VALUES:
        return MARKET_VALUES[clean_name] * 1_000_000
    
    # Try alias lookup
    canonical_name = TEAM_ALIASES.get(clean_name)
    if canonical_name and canonical_name in MARKET_VALUES:
        return MARKET_VALUES[canonical_name] * 1_000_000
    
    # Try case-insensitive search
    lower_name = clean_name.lower()
    for name, value in MARKET_VALUES.items():
        if name.lower() == lower_name:
            return value * 1_000_000
    
    # Try partial matching for common patterns
    for alias, canonical in TEAM_ALIASES.items():
        if alias.lower() == lower_name:
            if canonical in MARKET_VALUES:
                return MARKET_VALUES[canonical] * 1_000_000
    
    # Log unknown team and return default
    logger.debug(f"Unknown team '{team_name}', using default value {DEFAULT_VALUE_MILLIONS}M")
    return DEFAULT_VALUE_MILLIONS * 1_000_000


def get_all_team_values() -> dict:
    """
    Get all team values as a dictionary.
    
    Returns:
        Dictionary mapping team names to values in Euros
    """
    return {name: value * 1_000_000 for name, value in MARKET_VALUES.items()}


def get_expected_rank_by_value(team_name: str, league_teams: list) -> int:
    """
    Calculate expected rank based on market value within a league.
    
    Args:
        team_name: The team name
        league_teams: List of team names in the same league
        
    Returns:
        Expected rank (1 = highest value team)
    """
    if not league_teams:
        return 10  # Default middle rank
    
    # Get values for all teams
    team_values = [(t, get_team_value(t)) for t in league_teams]
    
    # Sort by value descending (highest value = rank 1)
    team_values.sort(key=lambda x: x[1], reverse=True)
    
    # Find the team's rank
    team_value = get_team_value(team_name)
    for i, (t, v) in enumerate(team_values, start=1):
        if v == team_value:
            return i
    
    return len(league_teams) // 2  # Default to middle if not found
