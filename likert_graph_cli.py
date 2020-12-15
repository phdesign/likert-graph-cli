import math
import os

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from colour import Color
from matplotlib.colors import to_rgba

facecolor = "#f2f5fc"
agree_color = "#3e6386"
neutral_color = "#c7cdd0"
disagree_color = "#e26d34"


def blend_colors(num):
    """Generate blending from agree to disagree."""
    steps = math.floor(num / 2.0) + 1
    agreeable_colors = list(Color(agree_color).range_to(Color(neutral_color), steps))
    disagreeable_colors = list(Color(neutral_color).range_to(Color(disagree_color), steps))
    # If it's an odd number of steps then there's no neutral
    if (num % 2) == 0:
        agreeable_colors.pop()
    return [c.hex_l for c in (agreeable_colors + disagreeable_colors[1:])]


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
            negative_comparison.plot(
                kind="barh", legend=False, align="center", width=0.2, ax=axis, color=disagree_color
            )
        if not positive_comparison.empty:
            positive_comparison.plot(kind="barh", legend=False, align="center", width=0.2, ax=axis, color=agree_color)
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
        text_color = disagree_color if width < 0 else agree_color
        axis.text(
            rec.get_x(),
            rec.get_y() - rec.get_height(),
            "{:.0f}".format(width * 100),
            ha="center",
            va="center",
            color=text_color,
        )


def plot_results(df, axis, title):
    df = df.drop(columns=["comparison", "agreeable", "agreeable_compare"], errors="ignore")
    df.plot(
        kind="barh",
        stacked=True,
        color=blend_colors(df.shape[1]),
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


def create_figure(subplot_rows, height, height_ratios, width_ratios, title):
    fig = plt.figure(figsize=(10, height))
    gs = fig.add_gridspec(
        nrows=subplot_rows,  # Two columns, one for results, one for comparison
        ncols=2,
        hspace=0.5,  # Vertical space between subplots
        wspace=0,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
    )
    axes = gs.subplots(sharey="row")
    fig.subplots_adjust(top=0.95)
    fig.suptitle(title, fontsize=18)
    return fig, axes


def create_graph(df, title):
    # Height is calculated from number of questions + groups,
    # this adjusts the weight of the height
    height_adjustment = 0.8
    question_count = df.shape[0]
    width_ratios = [0.8, 0.2] if "comparison" in df else [1, 0]

    if "_group" in df.index.names:
        groups = df.groupby("_group", dropna=False)
        fig, axes = create_figure(
            subplot_rows=groups.ngroups,
            height=(question_count + groups.ngroups) * height_adjustment,
            height_ratios=groups.size().tolist(),
            width_ratios=width_ratios,
            title=title,
        )
        for (key, group), i in zip(groups, range(groups.ngroups)):
            group = group.reset_index(level="_group", drop=True)
            plot_results(group, axes[i][0], key)
            plot_comparison(group, axes[i][1])
    else:
        fig, axes = create_figure(
            subplot_rows=1,
            height=question_count * height_adjustment,
            height_ratios=[1],
            width_ratios=width_ratios,
            title=title,
        )
        plot_results(df, axes[0], None)
        plot_comparison(df, axes[1])

    return fig


def set_graph_style():
    plt.style.use("default")
    mpl.rc("axes", facecolor=facecolor)
    mpl.rc("figure", facecolor=facecolor)
    mpl.rc("axes.spines", left=False, bottom=False, top=False, right=False)
    mpl.rc("xtick.major", size=0)
    mpl.rc("ytick.major", size=0)


def sample_data(output):
    columns = pd.MultiIndex.from_tuples(
        [
            ("", "Team"),
            ("Leadership", "The leaders at my company keep people informed about what is happening"),
            ("Leadership", "My manager is a great role model for employees"),
            ("Leadership", "The leaders at my company have communicated a vision that motivates me"),
            ("Enablement", "I have access to the things I need to do my job well"),
            ("Enablement", "I have access to the learning and development I need to do my job well"),
            ("Enablement", "Most of the systems and processes here support us getting our work done effectively"),
            ("Alignment", "I know what I need to do to be successful in my role"),
            ("Alignment", "I receive appropriate recognition when I do good work"),
            ("Alignment", "Day-to-day decisions here demonstrate that quality and improvement are top priorities"),
            (
                "Development",
                "My manager (or someone in management) has shown a genuine interest in my career aspirations",
            ),
            ("Development", "I believe there are good career opportunities for me at this company"),
            ("Development", "This is a great company for me to make a contribution to my development"),
        ]
    )
    df = pd.DataFrame(columns=columns)
    for i in range(100):
        df.loc[i] = np.append(
            np.random.choice(["Product Engineering", "Data Engineering", "Leadership", "Customer Experience"], 1),
            np.random.choice(["strongly agree", "agree", "neutral", "disagree", "strongly disagree"], size=(df.shape[1] - 1)),
            # np.random.randint(1, 5, size=(df.shape[1] - 1)),
        )
    df.to_csv(output, index=False)


def calc_percentages(df, group_level, compare=None):
    df = df.groupby(level=group_level).sum()
    # Convert counts into percentages
    df = df.div(df.sum(axis=1), axis=0)
    # Calculate agreeable score (sum of positive responses)
    df["agreeable"] = df.iloc[:, 0:2].sum(axis=1)
    # Reverse order of columns
    # response_percent = response_percent.iloc[:, ::-1]

    if compare is not None:
        df = df.reset_index(level=0).join(compare[["agreeable"]], rsuffix="_compare")
        df["comparison"] = df["agreeable"] - df["agreeable_compare"]

    return df


def pivot_questions(df, cohort_column, header_rows, index_names):
    if cohort_column is not None:
        if type(df.columns) == pd.MultiIndex:
            cohort_column_multiindex = df.columns[df.columns.get_level_values(1) == cohort_column][0]
        else:
            cohort_column_multiindex = cohort_column
        # Save cohort column to index
        df = df.set_index(cohort_column_multiindex).rename_axis("_cohort")
    # Filter columns to just numeric
    # df = df.select_dtypes(include="number")

    # Stack df -> count question by value
    df = pd.get_dummies(df.stack(header_rows))
    df = df.rename_axis(index_names)
    # Remove the record index when no cohort
    if cohort_column is None:
        df = df.reset_index(level=0, drop=True)
    return df


@click.command()
@click.argument("_input", metavar="INPUT", type=click.File("rb"))
@click.argument("output")
@click.option("-c", "--cohort-column")
@click.option("-g", "--has-groups", is_flag=True)
@click.option("-s", "--sample", is_flag=True)
def main(_input, output, cohort_column, has_groups, sample):
    value_order = ["strongly agree", "agree", "neutral", "disagree", "strongly disagree"]
    # TODO: Option to alphabetise the question / group order

    if sample:
        sample_data(output)
        exit()

    # Global figure styles
    set_graph_style()

    # Forking code
    if has_groups:
        header_rows = [0, 1]
    else:
        header_rows = [0]

    if has_groups:
        index_names = ["_cohort", "_group", "_question"]
    else:
        index_names = ["_cohort", "_question"]

    if has_groups:
        aggregate_group_by = [1, 2]
    elif cohort_column is not None:
        aggregate_group_by = [1]
    else:
        aggregate_group_by = [0]

    if cohort_column:
        aggregate_title = "All"
    else:
        aggregate_title = None

    if has_groups:
        results_group_by = [0, 1, 2]
    elif cohort_column is not None:
        results_group_by = [0, 1]
    else:
        results_group_by = None

    # Load the data
    results = pd.read_csv(_input, header=header_rows)
    print(f"total respondents: {results.shape[0]}")

    results = pivot_questions(results, cohort_column, header_rows, index_names)
    # Sort columns
    results = results[value_order]
    # print(results.head(n=20))
    # exit()

    # Create aggregate results
    aggregate = calc_percentages(results, aggregate_group_by)
    fig = create_graph(aggregate, aggregate_title)
    print(f"writing to {output}...")
    fig.savefig(output, bbox_inches="tight")

    if cohort_column is not None:
        # Count occurrences of responses
        results = calc_percentages(results, results_group_by, aggregate)
        (root, ext) = os.path.splitext(output)
        by_cohort = results.groupby("_cohort", dropna=False)
        for (key, group), i in zip(by_cohort, range(1, by_cohort.ngroups + 1)):
            group = group.drop(columns="_cohort")
            alt_output = f"{root}.{i}{ext}"

            fig = create_graph(group, key)
            print(f"writing {i} of {by_cohort.ngroups} to {alt_output}...")
            fig.savefig(alt_output, bbox_inches="tight")


if __name__ == "__main__":
    main()
