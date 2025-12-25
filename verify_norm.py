from src.data.odds_api_client import normalize_team

test_cases = [
    # Ligue 1
    ("Toulouse FC", "toulouse"),
    ("Racing Club de Lens", "lens"),
    ("AS Monaco FC", "monaco"),
    ("Olympique Lyonnais", "lyon"),
    ("OGC Nice", "nice"),
    ("RC Strasbourg Alsace", "strasbourg"),
    ("Lille OSC", "lille"),
    ("Stade Rennais FC 1901", "rennes"),
    ("Olympique de Marseille", "marseille"),
    ("FC Nantes", "nantes"),
    ("Le Havre AC", "le havre"),
    ("Angers SCO", "angers"),
    ("FC Lorient", "lorient"),
    ("FC Metz", "metz"),
    ("Stade Brestois 29", "brest"),
    ("AJ Auxerre", "auxerre"),
    ("Paris Saint-Germain FC", "psg"),
    ("Paris FC", "paris fc"),
    # La Liga
    ("Rayo Vallecano de Madrid", "rayo vallecano"),
    ("Getafe CF", "getafe"),
    ("RC Celta de Vigo", "celta vigo"),
    ("Valencia CF", "valencia"),
    ("CA Osasuna", "osasuna"),
    ("Athletic Club", "athletic bilbao"),
    ("Elche CF", "elche"),
    ("Villarreal CF", "villarreal"),
    ("RCD Espanyol de Barcelona", "espanyol"),
    ("FC Barcelona", "barcelona"),
    ("Sevilla FC", "sevilla"),
    ("Levante UD", "levante"),
    ("Real Madrid CF", "real madrid"),
    ("Real Betis Balompié", "real betis"),
    ("Deportivo Alavés", "alaves"),
    ("Real Oviedo", "oviedo"),
    ("RCD Mallorca", "mallorca"),
    ("Girona FC", "girona"),
    ("Real Sociedad de Fútbol", "real sociedad"),
    ("Club Atlético de Madrid", "atletico madrid")
]

print("--- VERIFYING NORMALIZATION ---")
failed = 0
for raw, expected in test_cases:
    actual = normalize_team(raw)
    if actual != expected:
        print(f"FAILED: '{raw}' -> Expected '{expected}', got '{actual}'")
        failed += 1
    else:
        print(f"OK: '{raw}' -> '{actual}'")

print(f"\nTotal: {len(test_cases)}, Failed: {failed}")
