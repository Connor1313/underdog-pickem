from datetime import datetime
import re
import requests
import bs4
import pandas as pd
import numpy as np

# Columns for our dataset (including odds columns)
vlrgg_columns = [
    "match-datetime", "team1", "team2", "team1-score", "team2-score",
    "team1-odds", "team2-odds",
    "player-name", "player-team", "agent", "rating", "acs", "k", "d", "a"
]

def vlrgg_scraper(url: str) -> pd.DataFrame:
    """
    Scraper to collect player statistics and pre-match odds from a VLR.gg match page.
    Pre-match odds are extracted from the page and the losing team's odds are computed
    using implied probabilities (both rounded to two decimal places).
    
    The player stats are parsed by splitting the row text on whitespace. If the resulting
    token list has 15 or more tokens, we assume the table has been split into T, CT, and average
    columns. In that case, we take the first group (T side) values:
      - rating  = token[2]
      - ACS     = token[5]
      - kills   = token[8]
      - deaths  = token[11]
      - assists = token[14]
    Otherwise, we fall back to previous logic.
    """
    page = requests.get(url)
    soup = bs4.BeautifulSoup(page.content, "html.parser")
    data_table = []
    
    # --- Extract match-level info: datetime ---
    match_stats = soup.find("div", {"class": "match-header-date"})
    if not match_stats:
        return pd.DataFrame()
    datetime_stat = match_stats.find("div", {"class": "moment-tz-convert"}).get("data-utc-ts")
    try:
        match_datetime = datetime.strptime(datetime_stat, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return pd.DataFrame()

    # --- Extract pre-match odds (if available) using the provided HTML snippet ---
    bet_elem = soup.find("a", class_="match-bet-item")
    if bet_elem:
        bet_div = bet_elem.find("div", class_="match-bet-item-team")
        # Extract the winning team's name from the odds block.
        winner_team = None
        winner_team_elem = bet_div.find("span", 
                                        style=lambda value: value and "font-weight: 700" in value)
        if winner_team_elem:
            winner_team = winner_team_elem.get_text(strip=True)
        # Extract the decimal odds value (e.g., "at 1.52 pre-match odds")
        odds_text = bet_div.get_text(" ", strip=True)
        odds_match = re.search(r"at\s+(\d+(?:\.\d+)?)\s+pre-match odds", odds_text)
        if odds_match:
            winner_odds_value = float(odds_match.group(1))
        else:
            winner_odds_value = np.nan
    else:
        winner_team = None
        winner_odds_value = np.nan

    # --- Calculate losing team's odds using implied probabilities ---
    if pd.notna(winner_odds_value) and winner_odds_value > 1:
        implied_prob_winner = 1 / winner_odds_value
        implied_prob_loser = 1 - implied_prob_winner
        if implied_prob_loser > 0:
            loser_odds_value = 1 / implied_prob_loser
        else:
            loser_odds_value = np.nan
    else:
        loser_odds_value = np.nan

    # --- Round odds to two decimal places ---
    if pd.notna(winner_odds_value):
        winner_odds_value = round(winner_odds_value, 2)
    if pd.notna(loser_odds_value):
        loser_odds_value = round(loser_odds_value, 2)

    # --- Extract team names, scores, and player stats for each map ---
    map_stats = soup.find_all("div", {"class": "vm-stats-game"})
    # Remove the overall ("all games") section.
    map_stats = [ms for ms in map_stats if ms.get("data-game-id") != "all"]

    for map_stat in map_stats:
        map_result = map_stat.find("div", {"class": "vm-stats-game-header"})
        # Get team names and scores.
        team_stats = map_result.find_all("div", {"class": "team"})
        if len(team_stats) < 2:
            continue  # Skip if both teams are not found.
        team1_stat, team2_stat = team_stats
        team1_name = team1_stat.find("div", {"class": "team-name"}).get_text(strip=True)
        team2_name = team2_stat.find("div", {"class": "team-name"}).get_text(strip=True)
        team1_score = team1_stat.find("div", {"class": "score"}).get_text(strip=True)
        team2_score = team2_stat.find("div", {"class": "score"}).get_text(strip=True)
        
        try:
            team1_score_val = int(team1_score)
            team2_score_val = int(team2_score)
        except Exception:
            continue  # Skip this map if scores are not valid numbers

        # Determine which team won and assign odds accordingly.
        if team1_score_val > team2_score_val:
            # team1 won.
            if winner_team and team1_name == winner_team:
                team1_odds = winner_odds_value
                team2_odds = loser_odds_value
            else:
                team1_odds = winner_odds_value
                team2_odds = loser_odds_value
        else:
            # team2 won.
            if winner_team and team2_name == winner_team:
                team2_odds = winner_odds_value
                team1_odds = loser_odds_value
            else:
                team2_odds = winner_odds_value
                team1_odds = loser_odds_value

        # --- Get the two tables for player stats (one for each team) ---
        teams_player_stats = map_stat.find_all("table", "wf-table-inset")
        if len(teams_player_stats) < 2:
            continue
        team1_player_stats, team2_player_stats = teams_player_stats

        def parse_player_stats(player_stats_table: bs4.element.Tag, player_team: str) -> None:
            tbody = player_stats_table.find("tbody")
            if not tbody:
                return
            rows = tbody.find_all("tr")
            for row in rows:
                # Extract the agent from the "mod-agents" cell.
                try:
                    agent = row.find("td", {"class": "mod-agents"}).find("img").get("title")
                except Exception:
                    agent = np.nan

                # Split the row text on any whitespace and filter out "/" tokens.
                raw_stats = [s for s in re.split(r"\s+", row.get_text()) if s and s != "/"]

                # If we have 15 or more tokens, assume the table is split into T, CT, and Average.
                # In that case, take the first (T side) values:
                if len(raw_stats) >= 15:
                    player_name = raw_stats[0]
                    rating = raw_stats[2]
                    acs = raw_stats[5]
                    k_val = raw_stats[8]
                    d_val = raw_stats[11]
                    a_val = raw_stats[14]
                else:
                    # Fallback: if not enough tokens, try a basic mapping.
                    if len(raw_stats) < 8:
                        continue  # Not enough tokens to parse stats
                    try:
                        float(raw_stats[3])
                        stats_start = 3
                    except ValueError:
                        stats_start = 4
                    if len(raw_stats) > stats_start + 6 and raw_stats[stats_start + 3] == "/":
                        rating = raw_stats[stats_start]
                        acs = raw_stats[stats_start + 1]
                        kills = raw_stats[stats_start + 2]
                        deaths = raw_stats[stats_start + 4]
                        if len(raw_stats) > stats_start + 6 and raw_stats[stats_start + 5] == "/":
                            assists = raw_stats[stats_start + 6]
                        else:
                            assists = raw_stats[stats_start + 5]
                        k_val = kills
                        d_val = deaths
                        a_val = assists
                    else:
                        rating = raw_stats[stats_start]
                        acs = raw_stats[stats_start+1]
                        k_val = raw_stats[stats_start+2]
                        d_val = raw_stats[stats_start+3]
                        a_val = raw_stats[stats_start+4]
                    player_name = raw_stats[0]
                # Build the row: match info, team info, odds, and player stats.
                row_data = [
                    match_datetime,
                    team1_name, team2_name, team1_score, team2_score,
                    team1_odds, team2_odds,
                    player_name, player_team, agent,
                    rating, acs, k_val, d_val, a_val
                ]
                data_table.append(row_data)

        # Process player rows for each team.
        parse_player_stats(team1_player_stats, team1_name)
        parse_player_stats(team2_player_stats, team2_name)

    df = pd.DataFrame(data_table, columns=vlrgg_columns)
    return df

def get_all_vlrgg_url(n_start: int, n_pages: int):
    """
    Returns a list of match URLs from VLR.gg result pages.
    """
    match_url_list = []
    prefix = "https://www.vlr.gg"
    for i in range(n_start, n_pages):
        match_result_page = f"{prefix}/matches/results/?page={i}"
        page = requests.get(match_result_page)
        soup = bs4.BeautifulSoup(page.content, "html.parser")
        match_cards = soup.find_all("div", {"class": "wf-card"})
        for match_card in match_cards:
            for url_element in match_card.find_all("a", {"class": "wf-module-item"}):
                match_url_list.append(prefix + url_element.get("href"))
    return match_url_list

if __name__ == "__main__":
    stop_scraping = False
    # Set cutoff date to June 25, 2024.
    cutoff_date = datetime(2024, 6, 25)
    
    all_dfs = []  # To accumulate all scraped data across batches.
    l1 = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280]
    for i in range(len(l1) - 1):
        url_list = get_all_vlrgg_url(l1[i], l1[i + 1])
        df_list = []
        error_count = 0
        count = 0
        for url in url_list:
            if count % 50 == 0:
                print(f"Processed {count} matches...")
            try:
                df = vlrgg_scraper(url)
                if not df.empty:
                    if df["match-datetime"].iloc[0] < cutoff_date:
                        stop_scraping = True
                        break
                    df_list.append(df)
            except Exception as e:
                error_count += 1
            count += 1
        if df_list:
            df_total = pd.concat(df_list, ignore_index=True)
            df_total.to_csv(f"batch_{l1[i]}-{l1[i+1]}.csv", index=False)
            all_dfs.append(df_total)
        else:
            print(f"No valid data found in batch {l1[i]}-{l1[i+1]}")
        print(f"Errors encountered in this batch: {error_count}")
        if stop_scraping:
            print("Encountered a match before June 25, 2024. Stopping further scraping.")
            break
    
    if all_dfs:
        full_data = pd.concat(all_dfs, ignore_index=True)
        full_data = full_data.sample(frac=1, random_state=42).reset_index(drop=True)
        train_size = int(0.75 * len(full_data))
        train_data = full_data.iloc[:train_size]
        backtest_data = full_data.iloc[train_size:]
        
        train_data.to_csv("train_data.csv", index=False)
        backtest_data.to_csv("backtest_data.csv", index=False)
        print(f"Total records: {len(full_data)}. Training: {len(train_data)}, Backtesting: {len(backtest_data)}.")
    else:
        print("No data was scraped.")
