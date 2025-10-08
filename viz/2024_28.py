import os
import math

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "2024", "week-28", "Superstore_with_LAT_LNG.xlsx"
)


def subplot_idx(i: int, num_cols: int) -> (int, int):
    """
    given a flat index of a subplot and a number of columns,
    this returns row/col indices for that subplot in a Plotly Figure

    Plotly subplots are 1-indexed
    """
    row = math.ceil((i + 1e-2) / num_cols)
    col = i % num_cols + 1
    return row, col


# ['Row ID', 'Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 'Customer ID', 'Customer Name', 'Segment', 'Country/Region', 'City', 'State/Province', 'Postal Code', 'Region', 'Product ID', 'Category', 'Sub-Category', 'Product Name', 'Sales', 'Quantity', 'Discount', 'Profit', 'LAT', 'LNG']


def main():
    df = pl.read_csv(DATA_PATH)

    # 59 unique State/Province

    states = df["State/Province"].unique().sort()
    num_cols = 10
    fig = make_subplots(
        rows=math.ceil(len(states) / num_cols),
        cols=num_cols,
        subplot_titles=states,
        horizontal_spacing=None,
        vertical_spacing=None,
        shared_yaxes=True,
        shared_xaxes=True,
    )

    pivoted_profit = df.pivot(
        index="State/Province", on="Category", values="Profit", aggregate_function="sum"
    ).fill_null(0)
    pivoted_sales = df.pivot(
        index="State/Province", on="Category", values="Sales", aggregate_function="sum"
    ).fill_null(0)

    # the largest in the data is -52% + 49%
    pct_clamp = 50

    for i, state in enumerate(states):
        r, c = subplot_idx(i, num_cols)
        state_profit = (
            pivoted_profit.filter(pl.col("State/Province") == state)
            .unpivot(
                on=pivoted_profit.columns[1:],
                variable_name="Category",
                value_name="Profit",
            )
            .sort("Category")
        )
        state_sales = (
            pivoted_sales.filter(pl.col("State/Province") == state)
            .unpivot(
                on=pivoted_sales.columns[1:],
                variable_name="Category",
                value_name="Sales",
            )
            .sort("Category")
        )
        state_df = state_profit.join(state_sales, on="Category").with_columns(
            (100 * pl.col("Profit") / pl.col("Sales")).fill_nan(0).round(0).alias("Profit Pct")
        ).sort("Category", descending=True)

        bar_colors = []
        for profit_pct in state_df["Profit Pct"]:
            clamped = min(pct_clamp, profit_pct)
            clamped = max(-pct_clamp, clamped)

            scaled = (clamped + pct_clamp) / (2 * pct_clamp)

            color_sample = sample_colorscale("Picnic_r", (scaled,))[0]
            bar_colors.append(color_sample)

        fig.add_trace(
            go.Bar(
                x=state_df["Profit Pct"],
                y=state_df["Category"],
                orientation="h",
                hovertemplate="%{y}: %{x}%<extra></extra>",
                marker_color=bar_colors,
            ),
            row=r,
            col=c,
        )

    fig.update_layout(
        template='plotly_dark',
        showlegend=False,
        title_text="Profit as a percentage of sales in each state/province",
    )
    fig.update_xaxes(range=[-pct_clamp, pct_clamp])
    fig.show()

    # TODO maybe add an "Overall" plot to fill out the full 60
    # so that it shows the x axis ticks on the last column

if __name__ == "__main__":
    main()
