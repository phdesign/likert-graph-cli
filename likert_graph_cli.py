import os

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

facecolor = "#f2f5fc"
series_colors = ["#466384", "#869caf", "#e4af8e", "#d67242"]


def contrasting_text_color(color):
    """Calculate the contrasting text colour.

    Uses euclidean distance between the facecolor and text colours
    to determine if black or white is more contrasting.
    """
    color = np.array(color)
    white = np.array(to_rgba("white"))
    black = np.array(to_rgba("black"))
    white_dist = np.linalg.norm(color - white)
    black_dist = np.linalg.norm(color - black)
    return white if white_dist > black_dist else black


def plot_comparison(df, axis):
    """Plot the comparison chart to the axis.

    Color of the bar and text depends on whether it's agreeable or disagreeable.
    """
    if "comparison" in df:
        # Separate negative and positive values so we can colour them differently
        negative_comparison = df.mask(df["comparison"].ge(0), other=np.nan)[["comparison"]]
        positive_comparison = df.mask(~df["comparison"].ge(0), other=np.nan)[["comparison"]]
        if not negative_comparison.empty:
            negative_comparison.plot(kind="barh", legend=False, align="center", width=0.2, ax=axis, color=series_colors[-1])
        if not positive_comparison.empty:
            positive_comparison.plot(kind="barh", legend=False, align="center", width=0.2, ax=axis, color=series_colors[0])
    axis.invert_yaxis()
    # Hide the grid and spine lines
    axis.axis("off")
    # Center chart around 0
    axis.set_xlim(-1, 1)
    # Display value
    for rec in axis.patches:
        width = rec.get_width()
        if width == 0:
            continue
        text_color = series_colors[-1] if width < 0 else series_colors[0]
        axis.text(
            rec.get_x(),
            rec.get_y() - rec.get_height(),
            "{:.0f}".format(width * 100),
            ha="center",
            va="center",
            color=text_color,
            )


def plot_results(df, axis, title):
    df.drop(columns=["comparison", "agreeable", "agreeable_all"], errors="ignore")\
        .plot(
            kind="barh",
            stacked=True,
            color=series_colors,
            legend=True,
            xlabel="",
            fontsize=10,
            ax=axis,
            title=title,
            edgecolor=facecolor,
        )
    # Add padding between question & graph
    axis.tick_params(axis="y", which="major", pad=30)
    # Put questions up the right way
    axis.invert_yaxis()
    # Show vertical grid bars
    axis.grid(True, axis="x")
    # Set vertical grid at 25% increments
    start, end = axis.get_xlim()
    axis.xaxis.set_ticks(np.arange(start, end, 0.25))
    # Convert fractional values (0.1) to percentages (10%)
    axis.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    # Set the location of the legend (default=best)
    axis.legend(loc="best")
    # Add value text on top of bar
    for rec in axis.patches:
        # Determine the best text colour to use
        text_color = contrasting_text_color(rec.get_facecolor())
        width = rec.get_width()
        if width == 0:
            continue
        axis.text(
            rec.get_x() + width / 2,
            rec.get_y() + rec.get_height() / 2,
            "{:.0f}%".format(width * 100),
            ha="center",
            va="center",
            color=text_color,
            )


def generate_graph(df, title):
    # Two columns, one for results, one for comparison
    ncols = 2
    # A row per group
    nrows = df.ngroups
    fig = plt.figure(figsize=(10, 30))
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        hspace=0.5,  # Vertical space between subplots
        wspace=0,
        height_ratios=df.size().tolist(),
        width_ratios=([0.8, 0.2] if "comparison" in df.count() else [1, 0]),
    )
    axes = gs.subplots(sharey="row")
    fig.subplots_adjust(top=0.95)
    fig.suptitle(title, fontsize=18)

    # Draw subplot for each group
    for (key, group), i in zip(df, range(df.ngroups)):
        group = group.reset_index(level="group", drop=True)
        plot_results(group, axes[i][0], key)
        plot_comparison(group, axes[i][1])

    return fig


def set_graph_style():
    plt.style.use("default")
    mpl.rc("axes", facecolor=facecolor)
    mpl.rc("figure", facecolor=facecolor)
    mpl.rc("axes.spines", left=False, bottom=False, top=False, right=False)
    mpl.rc("xtick.major", size=0)
    mpl.rc("ytick.major", size=0)


@click.command()
@click.argument("_input", metavar="INPUT", type=click.File("rb"))
@click.argument("output")
def main(_input, output):
    cohort_column = "What Team are you currently working on?"
    value_order = [1, 2, 3, 4]
    has_question_groups = True

    # Load the data
    header = [0, 1] if has_question_groups else [0]
    results = pd.read_csv(_input, header=header)
    print(f"total respondents: {results.shape[0]}")

    # Filter columns to just numeric
    results = (
        results.set_index([('Unnamed: 2_level_0', cohort_column)])
        .rename_axis('cohort')
        .select_dtypes(include="number")
    )
    # Stack results -> count question by value
    results = pd.get_dummies(results.stack(header)).rename_axis(["cohort", "group", "question"])

    # Create aggregate results
    aggregate = results.groupby(level=[1, 2]).sum()
    # Convert counts into percentages
    aggregate = aggregate.div(aggregate.sum(axis=1), axis=0)
    # Calculate agreeable score (sum of postitive responses)
    aggregate["agreeable"] = aggregate.iloc[:, 0:2].sum(axis=1)

    # Count occurrences of responses
    results = results.groupby(level=[0, 1, 2]).sum()
    # Convert counts into percentages
    results = results.div(results.sum(axis=1), axis=0)
    # Calculate agreeable score (sum of positive responses)
    results["agreeable"] = results.iloc[:, 0:2].sum(axis=1)
    # Reverse order of columns
    # response_percent = response_percent.iloc[:, ::-1]

    results = results.reset_index(level=0).join(aggregate[["agreeable"]], rsuffix="_all")
    results["comparison"] = results["agreeable"] - results["agreeable_all"]

    # Global figure styles
    set_graph_style()

    aggregate_by_group = aggregate.groupby("group", dropna=False)
    fig = generate_graph(aggregate_by_group, "All")
    print(f"writing to {output}...")
    fig.savefig(output, bbox_inches="tight")

    (root, ext) = os.path.splitext(output)
    by_cohort = results.groupby("cohort", dropna=False)
    for (key, group), i in zip(by_cohort, range(1, by_cohort.ngroups + 1)):
        group = group.drop(columns="cohort")
        alt_output = f"{root}.{i}{ext}"
        by_group = group.groupby("group", dropna=False)

        fig = generate_graph(by_group, key)
        print(f"writing {i} of {by_cohort.ngroups} to {alt_output}...")
        fig.savefig(alt_output, bbox_inches="tight")


if __name__ == "__main__":
    main()
