import os
from supabase import create_client, Client
from typing import Dict, List, Optional, Any
import logging
from ..config import get_settings

logger = logging.getLogger(__name__)

class SupabaseClient:
    def __init__(self):
        settings = get_settings()
        self.url = settings.supabase_url
        self.key = settings.supabase_key
        if not self.url or not self.key:
            logger.warning("Supabase URL or Key missing in configuration")
            self.client = None
        else:
            self.client: Client = create_client(self.url, self.key)

    def save_bet(self, stake: float, total_odds: float, potential_win: float, selections: List[Dict]) -> str:
        """
        Saves a bet and its selections to Supabase.
        """
        if not self.client:
            raise Exception("Supabase client not initialized")

        # Insert parent bet
        bet_data = {
            "stake": stake,
            "total_odds": total_odds,
            "potential_win": potential_win,
            "status": "pending"
        }
        res = self.client.table("bets").insert(bet_data).execute()
        bet_id = res.data[0]["id"]

        # Insert selections
        selection_data = []
        for sel in selections:
            selection_data.append({
                "bet_id": bet_id,
                "fixture_id": sel["fixture_id"],
                "market": sel["market"],
                "selection": sel["selection"],
                "odds": sel["odds"],
                "home_team": sel.get("home_team"),
                "away_team": sel.get("away_team"),
                "league_name": sel.get("league_name"),
                "status": "pending"
            })
        
        self.client.table("bet_selections").insert(selection_data).execute()
        return bet_id

    def get_all_bets(self, limit: int = 50) -> List[Dict]:
        """
        Fetches all bets with their associated selections.
        """
        if not self.client:
            return []

        # Fetch bets with nested selections
        try:
            res = self.client.table("bets").select("*, bet_selections(*)").order("id", desc=True).limit(limit).execute()
            return res.data
        except Exception as e:
            logger.error(f"Error fetching bets: {e}")
            return []

    def get_pending_selections(self) -> List[Dict]:
        if not self.client:
            return []
        res = self.client.table("bet_selections").select("*").eq("status", "pending").execute()
        return res.data

    def update_selection_result(self, selection_id: str, status: str, result: str, actual_score: str):
        if not self.client:
            return
        self.client.table("bet_selections").update({
            "status": status,
            "result": result,
            "actual_score": actual_score
        }).eq("id", selection_id).execute()

    def get_team_stats(self, team_id: int) -> Optional[Dict]:
        if not self.client:
            return None
        res = self.client.table("team_stats").select("*").eq("team_id", team_id).execute()
        if res.data:
            return res.data[0]
        return None

    def update_team_stats(self, team_id: int, team_name: str, stats: Dict):
        if not self.client:
            return
        data = {
            "team_id": team_id,
            "team_name": team_name,
            "stats": stats,
            "last_updated": "now()"
        }
        self.client.table("team_stats").upsert(data).execute()

# Singleton
_supabase_client = None

def get_supabase_client():
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = SupabaseClient()
    return _supabase_client
