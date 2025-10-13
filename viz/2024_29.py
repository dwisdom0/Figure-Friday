import os
import polars as pl
import plotly.express as px

from glicko2 import Player

from utils import get_week_path


def cast_to_int(df: pl.DataFrame, colname: str):
  return df.with_columns(pl.col(colname).str.replace_all(r"[^0-9]", "").cast(pl.Int32, strict=False))


def main():
  p = get_week_path(2024, 29)
  standings = pl.read_csv(os.path.join(p, "ewf_standings.csv"))
  print(standings.schema)
  print(standings)
  appearances = pl.read_csv(os.path.join(p, "ewf_appearances.csv"))
  appearances = cast_to_int(appearances, 'attendance')
  print(appearances.schema)
  print(appearances)
  matches = pl.read_csv(os.path.join(p, "ewf_matches.csv"))
  matches = cast_to_int(matches, 'attendance')
  matches = matches.with_columns(pl.col('date').str.strptime(pl.Datetime, "%Y-%m-%d"))
  print(matches.schema)
  print(matches)

  plot_data = matches.group_by(
      pl.col("date").dt.year().alias('year'),
      pl.col('date').dt.month().alias('month'),
      pl.col('tier')
    ).agg(
      pl.sum('attendance')
    ).with_columns(
      pl.concat_str([pl.col('year'), pl.lit('-'), pl.col('month')]).alias('year_month')
    ).sort(by=[
      pl.col('year_month'),
      pl.col('tier')
    ])

  fig = px.bar(plot_data,
    x='year_month', y='attendance', color='tier',
    color_discrete_sequence=px.colors.qualitative.Plotly
  )
  #fig.show()


  homefield_adv = matches.group_by(
      pl.col("date").dt.year().alias('year'),
      pl.col('date').dt.month().alias('month'),
  ).agg(
    ((pl.sum('home_team_win') / (pl.sum('home_team_win') + pl.sum('away_team_win')) - 0.5 ) * 2).alias('home_win_adv')
  ).with_columns(
    pl.concat_str([pl.col('year'), pl.lit('-'), pl.col('month')]).alias('year_month')
  )

  homefield_fig = px.bar(homefield_adv, x='year_month', y='home_win_adv')
  #homefield_fig.show()

  score_data = matches.group_by(
      pl.col("date").dt.year().alias('year'),
      pl.col('date').dt.month().alias('month'),
      pl.col('tier')
  ).agg(
    (pl.col('home_team_score') - pl.col('away_team_score')).sum().alias('home_score_adv')
  ).with_columns(
      pl.concat_str([pl.col('year'), pl.lit('-'), pl.col('month')]).alias('year_month')
  )

  score_fig = px.bar(score_data, x='year_month', y='home_score_adv', color='tier', barmode='group')
  #score_fig.show()


  # add a column that has the most recent name of a team
  # they change names sometimes but keep the same team_id
  # I only want one entry in the plot legend,
  # so I'm going to show the most recent name
  most_recent_name = appearances.select(
    [pl.col('team_id'), pl.col('date')]
  ).group_by(
    pl.col('team_id')
  ).agg(
    pl.col('date').max().alias('date')
  )
  most_recent_name = most_recent_name.join(
    appearances,
    on=[pl.col('team_id'), pl.col('date')]
  ).select([
    pl.col('team_id'),
    pl.col('team_name').alias('final_team_name')
  ])

  appearances = appearances.join(most_recent_name, on='team_id')
  matches = matches.join(
    most_recent_name,
    how='left',
    left_on='home_team_id',
    right_on='team_id',
    suffix='_home',
  ).join(
    most_recent_name,
    how='left',
    left_on='away_team_id',
    right_on='team_id',
    suffix='_away',
  ).with_columns([
    pl.col('final_team_name').alias('final_home_team_name'),
    pl.col('final_team_name_away').alias('final_away_team_name'),
  ])


  for tier in appearances['tier'].unique().to_numpy().tolist():
    players = {}
    for player_id in appearances.filter(pl.col('tier') == tier)['team_id'].unique():
      players[player_id] = Player()

    glicko_plot_data = []
    for match in matches.filter(pl.col('tier') == tier).iter_rows(named=True):
      home_result = 0.5
      if match['home_team_win']:
        home_result = 1
      elif match['away_team_win']:
        home_result = 0

      home = players[match['home_team_id']]
      away = players[match['away_team_id']]
      glicko2_update(home, away, home_result)
      glicko_plot_data.append({'date': match['date'], 'team_id': match['home_team_id'], 'team_name': match['final_home_team_name'], 'rating': home.rating, 'RD': home.rd})
      glicko_plot_data.append({'date': match['date'], 'team_id': match['away_team_id'], 'team_name': match['final_away_team_name'], 'rating': away.rating, 'RD': away.rd})

    glicko_plot_df = pl.from_records(glicko_plot_data)

    print(f'\nTier {tier}')
    for player_id, player in players.items():
      print(f'Player {player_id}: Rating: {player.rating:.2f} RD: {player.rd:.2f}')

    rating_fig = px.line(glicko_plot_df, x='date', y='rating', color='team_name', title=f'Tier {tier} Glicko2')
    rating_fig.show()
  breakpoint()



def glicko2_update(p1: Player , p2: Player, p1_outcome: float):
  """
  outcome is from the perspective of p1
  1   = p1 won
  0.5 = draw
  0   = p2 won
  """
  if p1_outcome not in [0, 0.5, 1]:
    raise ValueError(f'p1_outcome must be one of [0, 0.5, 1]')
  p1_r_prev = p1.rating
  p1_rd_prev = p1.rd
  p2_r_prev = p2.rating
  p2_rd_prev = p2.rd

  p1.update_player([p2_r_prev], [p2_rd_prev], [p1_outcome])
  if p1_outcome == 0:
    p2_outcome = 1
  elif p1_outcome == 1:
    p2_outcome = 0
  else:
    p2_outcome= 0.5
  p2.update_player([p1_r_prev], [p1_rd_prev], [p2_outcome])




  # TODO:
  # Elo?
  # look into ways of ranking players with few bouts
  # qwen claims Elo is probably fine
  # TrueSkill by microsoft
  # Bayesian Elo
  # Bradley-Terry Model
  # Glicko2
  # Hierarchical Bayesian Models (Stan or PyMC3)

  # qwen reccomends glicko2 if Elo doesn't work
  # I kind of like Bradley-Terry for no good reason
  # but there's a python implementation of Glicko2
  # and bradley-terry is apparently just Elo


  # TODO:
  # unify Ladies / Women?
  # a few of the teams renamed themselvs from X Ladies to X Women
  # they have the same team_id so I guess it probably doesn't matter




if __name__ == "__main__":
  main()
