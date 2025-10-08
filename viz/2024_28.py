import os
import math

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    print(df.group_by(["Category", "Region"]).agg(pl.sum("Profit")).sort("Profit"))

    print(df.group_by(["Sub-Category", "Region"]).agg(pl.sum("Profit")).sort("Profit"))

    px.bar(
        df.group_by(["Sub-Category", "Region"]).agg(pl.sum("Profit")),
        "Sub-Category",
        "Profit",
        color="Region",
    ).show()

    px.bar(
        df.group_by(["Category", "State/Province"])
        .agg(pl.sum("Profit"))
        .sort("State/Province"),
        "State/Province",
        "Profit",
        color="Category",
        barmode="group",
    ).show()

    #px.bar(
    #    df.group_by(["Category", "State/Province"])
    #    .agg(pl.sum("Profit"))
    #    .sort("State/Province"),
    #    "Category",
    #    "Profit",
    #    facet_col="State/Province"
    #).show()

    # 59 unique State/Province

    states = df["State/Province"].unique().sort()
    num_cols = 10
    fig = make_subplots(
        rows=math.ceil(len(states) / num_cols), cols=num_cols, subplot_titles=states
    )
    for i, state in enumerate(states):
        r, c = subplot_idx(i, num_cols)
        state_df = (
            df.filter(pl.col("State/Province") == state)
            .group_by(["State/Province", "Category"])
            .agg(pl.sum("Profit"))
            .sort("Category")
        )

        fig.add_trace(
            go.Bar(x=state_df["Profit"], y=state_df["Category"], orientation="h"),
            row=r,
            col=c,
        )

    fig.update_layout(showlegend=False)
    fig.show()


if __name__ == "__main__":
    main()
