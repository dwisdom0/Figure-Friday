import os
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from glicko2 import Player

from utils import get_week_path


def cast_to_int(df: pl.DataFrame, colname: str):
    return df.with_columns(
        pl.col(colname).str.replace_all(r"[^0-9]", "").cast(pl.Int32, strict=False)
    )


def main():
    # testing NA fill behavior
    # seems to do what I want
    # x_data = list(range(10))
    # y_data = [1,2,3,4,5,None, None, None,9, 10]
    # test_data = {'x': x_data, 'y': y_data}
    # df = pl.DataFrame(test_data)

    # fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df['x'], y=df['y']))
    # fig.show()
    # breakpoint()

    p = get_week_path(2024, 29)
    standings = pl.read_csv(os.path.join(p, "ewf_standings.csv"))
    print(standings.schema)
    print(standings)
    appearances = pl.read_csv(os.path.join(p, "ewf_appearances.csv"))
    appearances = cast_to_int(appearances, "attendance")
    print(appearances.schema)
    print(appearances)
    matches = pl.read_csv(os.path.join(p, "ewf_matches.csv"))
    matches = cast_to_int(matches, "attendance")
    matches = matches.with_columns(pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d"))
    print(matches.schema)
    print(matches)

    plot_data = (
        matches.group_by(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            pl.col("tier"),
        )
        .agg(pl.sum("attendance"))
        .with_columns(
            pl.concat_str([pl.col("year"), pl.lit("-"), pl.col("month")]).alias(
                "year_month"
            )
        )
        .sort(by=[pl.col("year_month"), pl.col("tier")])
    )

    fig = px.bar(
        plot_data,
        x="year_month",
        y="attendance",
        color="tier",
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )
    # fig.show()

    homefield_adv = (
        matches.group_by(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        )
        .agg(
            (
                (
                    pl.sum("home_team_win")
                    / (pl.sum("home_team_win") + pl.sum("away_team_win"))
                    - 0.5
                )
                * 2
            ).alias("home_win_adv")
        )
        .with_columns(
            pl.concat_str([pl.col("year"), pl.lit("-"), pl.col("month")]).alias(
                "year_month"
            )
        )
    )

    homefield_fig = px.bar(homefield_adv, x="year_month", y="home_win_adv")
    # homefield_fig.show()

    score_data = (
        matches.group_by(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
            pl.col("tier"),
        )
        .agg(
            (pl.col("home_team_score") - pl.col("away_team_score"))
            .sum()
            .alias("home_score_adv")
        )
        .with_columns(
            pl.concat_str([pl.col("year"), pl.lit("-"), pl.col("month")]).alias(
                "year_month"
            )
        )
    )

    score_fig = px.bar(
        score_data, x="year_month", y="home_score_adv", color="tier", barmode="group"
    )
    # score_fig.show()

    # add a column that has the most recent name of a team
    # they change names sometimes but keep the same team_id
    # I only want one entry in the plot legend,
    # so I'm going to show the most recent name
    most_recent_name = (
        appearances.select([pl.col("team_id"), pl.col("date")])
        .group_by(pl.col("team_id"))
        .agg(pl.col("date").max().alias("date"))
    )
    most_recent_name = most_recent_name.join(
        appearances, on=[pl.col("team_id"), pl.col("date")]
    ).select([pl.col("team_id"), pl.col("team_name").alias("final_team_name")])

    appearances = appearances.join(most_recent_name, on="team_id")
    matches = (
        matches.join(
            most_recent_name,
            how="left",
            left_on="home_team_id",
            right_on="team_id",
            suffix="_home",
        )
        .join(
            most_recent_name,
            how="left",
            left_on="away_team_id",
            right_on="team_id",
            suffix="_away",
        )
        .with_columns(
            [
                pl.col("final_team_name").alias("final_home_team_name"),
                pl.col("final_team_name_away").alias("final_away_team_name"),
            ]
        )
    )

    # trying to get them from their website is a bit of a mess
    # switching to this
    # https://teamcolorcodes.com/england-national-football-team-color-codes/
    team_color_lkp = {
        "Reading Women": "#004494",
        "Tottenham Hotspur Women": "#132257",
        "Manchester United Women": "#da291c",
        "Doncaster Rovers Belles": "#ffd242",  # not on the teamcolorcodes website
        "Sunderland Women": "#dc0714",  # not on the teamcolorcodes website
        "Yeovil Town Ladies": "#00892f",  # not on the teamcolorcodes website
        "Everton Women": "#003399",
        "Manchester City Women": "#6cabdd",
        "Bristol City Women": "#e11f26",  # not on the teamcolorcodes website
        "Arsenal Women": "#ef0107",
        "West Ham United Women": "#7A263A",
        "Brighton and Hove Albion Women": "#0057b8",
        "Birmingham City Women": "#183a90",  # not on the teamcolorcodes website
        "Notts County Ladies": "#a3915f",  # not on the teamcolorcodes website
        "Chelsea Women": "#034694",
        "Aston Villa Women": "#95bfe5",
        "Leicester City Women": "#003090",
        "Liverpool Women": "#C8102e",
        "Blackburn Rovers Women": "#004D9C",  # not on the teamcolorcodes website
        "Coventry United Women": "#004D9C",  # not on the teamcolorcodes website
        "Sheffield Ladies": "#c61f41",  # not on the teamcolorcodes website, could also be black #000 since there are so many reds already
        "London Bees": "#ff7300",  # not on the teamcolorcodes website
        "Watford Women": "#fbee23",
        "Charlton Athletic Women": "#bf1829",  # not on the teamcolorcodes website, the logo is a gradient
        "Oxford United Women": "#fff200",  # not on the teamcolorcodes website
        "Southampton Women": "#d71920",
        "Durham Women": "#3858e9",  # not on the teamcolorcodes website
        "Sheffield United Women": "#ee2737",
        "Lewes Women": "#c6183d",  # not on the teamcolorcodes website, could also be black #000 or maybe yellow
        "London City Lionesses": "#00abc7",  # not on the teamcolorcodes website
        "Crystal Palace Women": "#1b458f",
    }

    BIG_FOUR = [
        "Arsenal Women",
        "Chelsea Women",
        "Manchester City Women",
        "Manchester United Women",
    ]

    rating_fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.02,
        shared_yaxes=True,
        shared_xaxes=True,
        subplot_titles=[
            "Women's Super League (Tier 1)",
            "Women's Super League 2 (Tier 2)",
        ],
    )

    legend_teams_shown = set()
    for tier in [1, 2]:
        players = {}
        for player_id in appearances.filter(pl.col("tier") == tier)["team_id"].unique():
            players[player_id] = Player()

        glicko_plot_data = []
        for match in matches.filter(pl.col("tier") == tier).iter_rows(named=True):
            home_result = 0.5
            if match["home_team_win"]:
                home_result = 1
            elif match["away_team_win"]:
                home_result = 0

            home = players[match["home_team_id"]]
            away = players[match["away_team_id"]]
            glicko2_update(home, away, home_result)
            glicko_plot_data.append(
                {
                    "date": match["date"],
                    "team_id": match["home_team_id"],
                    "team_name": match["final_home_team_name"],
                    "rating": home.rating,
                    "RD": home.rd,
                }
            )
            glicko_plot_data.append(
                {
                    "date": match["date"],
                    "team_id": match["away_team_id"],
                    "team_name": match["final_away_team_name"],
                    "rating": away.rating,
                    "RD": away.rd,
                }
            )

        glicko_plot_df = pl.from_records(glicko_plot_data)

        # getting some help from Qwen3
        # put NA values in the big gaps that happen
        # when a team gets promoted and then relegated
        # so that plotly won't connect the start and end
        # of those gaps
        glicko_plot_df = glicko_plot_df.with_columns(
            [
                pl.col("date").shift(1).over("team_id").alias("prev_date"),
                pl.col("date").over("team_id").alias("current_date"),
            ]
        ).with_columns(
            [
                # Compute difference in days
                (pl.col("current_date") - pl.col("prev_date"))
                .dt.total_days()
                .alias("days_diff"),
            ]
        )
        # 110 also breaks the gap between each season
        # gap_threshold_days = 110
        gap_threshold_days = 300
        glicko_plot_df = glicko_plot_df.with_columns(
            [
                pl.when(
                    (pl.col("days_diff").is_not_null())
                    & (pl.col("days_diff") > gap_threshold_days)
                )
                .then(True)
                .otherwise(False)
                .alias("is_gap")
            ]
        )
        glicko_plot_df = glicko_plot_df.with_columns(
            [
                pl.when(pl.col("is_gap"))
                .then(None)
                .otherwise(pl.col("rating"))
                .alias("glicko_rating_fixed")
            ]
        )

        # sort and then move the big 4 to the end
        # so they get drawn on top of everything else
        ordered_team_names = (
            glicko_plot_df["team_name"].unique().sort().to_numpy().tolist()
        )
        for t in BIG_FOUR:
            try:
                ordered_team_names.remove(t)
            except ValueError:
                continue
            ordered_team_names.append(t)

        print(f"\nTier {tier}")
        for team_name in ordered_team_names:
            player_id = glicko_plot_df.filter(pl.col("team_name") == team_name)[
                "team_id"
            ].unique()[0]
            player = players[player_id]
            print(
                f"Team: {team_name} Player {player_id}: Rating: {player.rating:.2f} RD: {player.rd:.2f}"
            )

            team_name = glicko_plot_df.filter(pl.col("team_id") == player_id)[
                "team_name"
            ].unique()[0]

            rating_fig.add_trace(
                go.Scatter(
                    x=glicko_plot_df.filter(pl.col("team_id") == player_id)["date"],
                    y=glicko_plot_df.filter(pl.col("team_id") == player_id)[
                        "glicko_rating_fixed"
                    ],
                    name=f"{team_name}",
                    legendgroup=team_name if team_name in BIG_FOUR else "Other",
                    legendgrouptitle_text=None if team_name in BIG_FOUR else "Other",
                    legendrank=1 if team_name in BIG_FOUR else 1000,
                    marker_color=team_color_lkp[team_name]
                    if team_name in BIG_FOUR
                    else "rgba(200, 200,200, 50)",
                    hovertemplate=f"(%{{y:.0f}}) {team_name} <extra></extra>",
                    # showlegend=False,
                    showlegend=team_name not in legend_teams_shown,
                ),
                row=tier,
                col=1,
            )
            legend_teams_shown.add(team_name)

    # default is margin=dict(l=80, r=80, t=100, b=80)
    rating_fig.update_layout(
        margin=dict(l=80, r=80, t=40, b=40),
        title_text="Manchester United closed the gap after a late start",
    )
    for r in [1, 2]:
        rating_fig.update_yaxes(
            range=[1000, 2000],
            title_text="Glicko-2 Rating",
            row=r,
            col=1,
        )
    rating_fig.show()

    # fig.write_html("figure_friday_2024_29.html", include_plotlyjs="cdn")


def glicko2_update(p1: Player, p2: Player, p1_outcome: float):
    """
    outcome is from the perspective of p1
    1   = p1 won
    0.5 = draw
    0   = p2 won
    """
    if p1_outcome not in [0, 0.5, 1]:
        raise ValueError(f"p1_outcome must be one of [0, 0.5, 1]")
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
        p2_outcome = 0.5
    p2.update_player([p1_r_prev], [p1_rd_prev], [p2_outcome])


if __name__ == "__main__":
    main()
