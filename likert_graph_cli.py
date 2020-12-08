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
            negative_comparison.plot(
                kind="barh", legend=False, align="center", width=0.2, ax=axis, color=series_colors[-1]
            )
        if not positive_comparison.empty:
            positive_comparison.plot(
                kind="barh", legend=False, align="center", width=0.2, ax=axis, color=series_colors[0]
            )
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
    df.drop(columns=["comparison", "agreeable", "agreeable_all"], errors="ignore").plot(
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
            np.random.randint(1, 5, size=(df.shape[1] - 1)),
        )
    df.to_csv(output, index=False)
    exit()


@click.command()
@click.argument("_input", metavar="INPUT", type=click.File("rb"))
@click.argument("output")
@click.option("-c", "--cohort-column")
@click.option("-g", "--has-groups", is_flag=True)
def main(_input, output, cohort_column, has_groups):
    value_order = [1, 2, 3, 4]
    # TODO: Option to alphabetise the question / group order

    # sample_data(output)

    # Global figure styles
    set_graph_style()

    # Load the data
    header_rows = [0, 1] if has_groups else [0]
    results = pd.read_csv(_input, header=header_rows)
    print(f"total respondents: {results.shape[0]}")

    if cohort_column is not None:
        # Save cohort column to index
        cohort_column_multiindex = (
            results.columns[results.columns.get_level_values(1) == cohort_column][0] if has_groups else cohort_column
        )
        results = results.set_index(cohort_column_multiindex).rename_axis("_cohort")
    # Filter columns to just numeric
    results = results.select_dtypes(include="number")

    # Stack results -> count question by value
    results = pd.get_dummies(results.stack(header_rows))
    index_names = ["_cohort", "_group", "_question"] if has_groups else ["_cohort", "_question"]
    results = results.rename_axis(index_names)
    # Remove the record index when no cohort
    if cohort_column is None:
        results = results.reset_index(level=0, drop=True)

    # Create aggregate results
    if has_groups:
        aggregate_group_by = [1, 2]
    elif cohort_column is not None:
        aggregate_group_by = [1]
    else:
        aggregate_group_by = [0]
    aggregate = results.groupby(level=aggregate_group_by).sum()
    # Convert counts into percentages
    aggregate = aggregate.div(aggregate.sum(axis=1), axis=0)
    # Calculate agreeable score (sum of positive responses)
    aggregate["agreeable"] = aggregate.iloc[:, 0:2].sum(axis=1)

    title = "All" if cohort_column is not None else None
    fig = create_graph(aggregate, title)
    print(f"writing to {output}...")
    fig.savefig(output, bbox_inches="tight")

    if cohort_column is None:
        return

    # Count occurrences of responses
    results_group_by = [0, 1, 2] if has_groups else [0, 1]
    results = results.groupby(level=results_group_by).sum()
    # Convert counts into percentages
    results = results.div(results.sum(axis=1), axis=0)
    # Calculate agreeable score (sum of positive responses)
    results["agreeable"] = results.iloc[:, 0:2].sum(axis=1)
    # Reverse order of columns
    # response_percent = response_percent.iloc[:, ::-1]

    results = results.reset_index(level=0).join(aggregate[["agreeable"]], rsuffix="_all")
    results["comparison"] = results["agreeable"] - results["agreeable_all"]

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
