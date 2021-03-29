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

face_color = "#f2f5fc"
agree_color = "#3e6386"
neutral_color = "#c7cdd0"
disagree_color = "#e26d34"

# Subplot height is calculated from the number of questions + groups,
# this constant adjusts the weight of those values
subplot_height_adjustment = 0.8
# This adjust the space at the top of the graph for the title, it's
# supposed to be in inches but it's inconistent
title_margin_adjustment = 1.2

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

    Uses euclidean distance between the face_color and text colours
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


def plot_results(df, axis, title, colors, show_legend):
    df = df.drop(columns=["comparison", "agreeable", "agreeable_compare"], errors="ignore")
    df.plot(
        kind="barh",
        stacked=True,
        color=colors,
        legend=show_legend,
        xlabel="",
        fontsize=10,
        ax=axis,
        title=title,
        edgecolor=face_color,
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
    fig.suptitle(title, fontsize=18)
    return fig, axes


def adjust_title_space(fig, margin):
    """Give the super title some breathing room."""
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    fig_height = (h - (1 - s.top) * h) + margin
    top = 1 - (margin / fig_height)
    bottom = s.bottom * (h / fig_height)
    fig.subplots_adjust(bottom=bottom, top=top)
    fig.set_figheight(fig_height)


def create_graph(df, title, colors, show_legend):
    question_count = df.shape[0]
    width_ratios = [0.8, 0.2] if "comparison" in df else [1, 0]

    if "_group" in df.index.names:
        groups = df.groupby("_group", dropna=False)
        fig, axes = create_figure(
            subplot_rows=groups.ngroups,
            height=(question_count + groups.ngroups) * subplot_height_adjustment,
            height_ratios=groups.size().tolist(),
            width_ratios=width_ratios,
            title=title,
        )
        for (key, group), i in zip(groups, range(groups.ngroups)):
            group = group.reset_index(level="_group", drop=True)
            plot_results(group, axes[i][0], key, colors, show_legend)
            plot_comparison(group, axes[i][1])
    else:
        fig, axes = create_figure(
            subplot_rows=1,
            height=question_count * subplot_height_adjustment,
            height_ratios=[1],
            width_ratios=width_ratios,
            title=title,
        )
        plot_results(df, axes[0], None, colors, show_legend)
        plot_comparison(df, axes[1])

    if title is not None:
        adjust_title_space(fig, title_margin_adjustment)

    return fig


def set_graph_style():
    plt.style.use("default")
    mpl.rc("axes", facecolor=face_color)
    mpl.rc("figure", facecolor=face_color)
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
    # Assume the first half of the columns are agreeable
    agreeable_columns = math.floor(df.shape[1] / 2.0)
    # Calculate agreeable score (sum of positive responses)
    df["agreeable"] = df.iloc[:, 0:agreeable_columns].sum(axis=1)

    if compare is not None:
        df = df.reset_index(level=0).join(compare[["agreeable"]], rsuffix="_compare")
        df["comparison"] = df["agreeable"] - df["agreeable_compare"]

    return df


def set_cohort_index(df, cohort_column):
    if type(df.columns) == pd.MultiIndex:
        cohort_column_multiindex = df.columns[df.columns.get_level_values(1) == cohort_column][0]
    else:
        cohort_column_multiindex = cohort_column
    # Save cohort column to index
    df = df.set_index(cohort_column_multiindex).rename_axis("_cohort")
    return df


def pivot_questions(df, header_rows, index_names):
    # Force all values to strings so stacking doesn't make them floats
    df = df.astype(str)
    # Stack df -> count question by value
    df = pd.get_dummies(df.stack(header_rows))
    df = df.rename_axis(index_names)
    return df


def sort_columns(df, value_order):
    if value_order is not None:
        # Use only the value items that exist in the data, otherwise we'll get an error
        # Convert all columns to strings (some may be numbers)
        existing_columns = [str(c) for c in df.columns.tolist()]
        df.columns = existing_columns
        filter_columns = [c for c in value_order if c in existing_columns]
        df = df[filter_columns]

    # Reverse order of columns
    # df = df.iloc[:, ::-1]
    return df


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.argument("_input", metavar="INPUT", type=click.File("rb"))
@click.argument("output")
@click.option("-c", "--cohort-column", help="Name of the column to group cohorts by. Will output multiple graphs.")
@click.option("-g", "--has-groups", is_flag=True, help="Expect INPUT to have first header row of column groups.")
@click.option("--legend/--no-legend", default=True, help="Show / hide legend on graph (defaults to show).")
@click.option("-n", "--numeric-only", is_flag=True, help="Filter columns to those that have numeric values only")
@click.option("-s", "--sample", is_flag=True, help="Generate a sample csv file of random data and exit.")
@click.option("-v", "--values", help="Comma-separated list of value names in order from positive to negative.")
def main(_input, output, cohort_column, has_groups, legend, numeric_only, sample, values):
    """Generates a horizonal bar graph based on likert scores (agree, disagree, etc...).

    INPUT expects a csv file of scores, one response per row with each question as a column.
    OUTPUT is the filename for the generated graph (png). If multiple graphs will be created (via cohorts), a number will be appended to the filename, e.g. output_1.png.
    """

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

    # Load the data
    results = pd.read_csv(_input, header=header_rows)
    print(f"total respondents: {results.shape[0]}")

    if cohort_column is not None:
        results = set_cohort_index(results, cohort_column)
        counts = results.index.value_counts()

    # Filter by columns with only numeric values
    if numeric_only:
        results = results.select_dtypes(include="number")

    results = pivot_questions(results, header_rows, index_names)
    # Remove the record index when no cohort
    if cohort_column is None:
        results = results.reset_index(level=0, drop=True)
    # Sort columns
    value_order = (
        [v for v in (w.strip() for w in values.split(',')) if v != ""]
        if values is not None else None
    )
    results = sort_columns(results, value_order)

    # If we have a list of values, use that to determine colors (allows for missing values in the data)
    colors = blend_colors(len(value_order) if value_order is not None else results.shape[1])

    if has_groups:
        aggregate_group_by = [1, 2]
    elif cohort_column is not None:
        aggregate_group_by = [1]
    else:
        aggregate_group_by = [0]

    if cohort_column:
        aggregate_title = f"All (n={counts.sum()})"
    else:
        aggregate_title = None

    # Create aggregate results
    aggregate = calc_percentages(results, aggregate_group_by)
    fig = create_graph(aggregate, aggregate_title, colors, legend)
    print(f"writing to {output}...")
    fig.savefig(output, bbox_inches="tight")

    if cohort_column:
        if has_groups:
            results_group_by = [0, 1, 2]
        else:
            results_group_by = [0, 1]

        # Count occurrences of responses
        results = calc_percentages(results, results_group_by, aggregate)
        (root, ext) = os.path.splitext(output)
        by_cohort = results.groupby("_cohort", dropna=False)
        for (key, group), i in zip(by_cohort, range(1, by_cohort.ngroups + 1)):
            group = group.drop(columns="_cohort")
            alt_output = f"{root}_{i}{ext}"
            title = f"{key} (n={counts.loc[key]})"

            fig = create_graph(group, title, colors, legend)
            print(f"writing {i} of {by_cohort.ngroups} to {alt_output}...")
            fig.savefig(alt_output, bbox_inches="tight")


if __name__ == "__main__":
    main()
