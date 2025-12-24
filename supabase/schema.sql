-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table for bets (parent)
CREATE TABLE IF NOT EXISTS bets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stake DECIMAL NOT NULL,
    total_odds DECIMAL NOT NULL,
    potential_win DECIMAL NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'won', 'lost', 'cancelled')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for bet selections (children)
CREATE TABLE IF NOT EXISTS bet_selections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    bet_id UUID REFERENCES bets(id) ON DELETE CASCADE,
    fixture_id INTEGER NOT NULL,
    home_team TEXT,
    away_team TEXT,
    market TEXT NOT NULL,
    selection TEXT NOT NULL,
    odds DECIMAL NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'won', 'lost', 'cancelled')),
    result TEXT,
    actual_score TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for team stats cache
CREATE TABLE IF NOT EXISTS team_stats (
    team_id INTEGER PRIMARY KEY,
    team_name TEXT,
    stats JSONB, -- stores goals_scored_avg, goals_conceded_avg, etc.
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trigger to update bet status based on selections
CREATE OR REPLACE FUNCTION update_bet_status()
RETURNS TRIGGER AS $$
BEGIN
    -- If any selection is 'lost', the whole bet is 'lost'
    IF (SELECT COUNT(*) FROM bet_selections WHERE bet_id = NEW.bet_id AND status = 'lost') > 0 THEN
        UPDATE bets SET status = 'lost' WHERE id = NEW.bet_id;
    -- If all selections are 'won', the whole bet is 'won'
    ELSIF (SELECT COUNT(*) FROM bet_selections WHERE bet_id = NEW.bet_id AND status = 'won') = 
          (SELECT COUNT(*) FROM bet_selections WHERE bet_id = NEW.bet_id) THEN
        UPDATE bets SET status = 'won' WHERE id = NEW.bet_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_bet_status ON bet_selections;
CREATE TRIGGER trg_update_bet_status
AFTER UPDATE OF status ON bet_selections
FOR EACH ROW
EXECUTE FUNCTION update_bet_status();
