import os

import click
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

facecolor = "#f2f5fc"


def generate_graph(df, title):
    colors = ["#466384", "#869caf", "#e4af8e", "#d67242"]

    def get_contrast_color(facecolor):
        """Calculate the contrasting text colour.

        Uses euclidean distance between the facecolor and text colours
        to determine if black or white is more contrasting.
        """
        facecolor = np.array(facecolor)
        white = np.array(to_rgba("white"))
        black = np.array(to_rgba("black"))
        white_dist = np.linalg.norm(facecolor - white)
        black_dist = np.linalg.norm(facecolor - black)
        return white if white_dist > black_dist else black

    # Calculate the number of subplots based on our groups
    ncols = 2
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
        ax = group[[1, 2, 3, 4]].plot(
            kind="barh",
            stacked=True,
            color=colors,
            legend=True,
            xlabel="",
            fontsize=10,
            ax=axes[i][0],
            title=key,
            edgecolor=facecolor,
        )
        # Add padding between question & graph
        ax.tick_params(axis="y", which="major", pad=30)
        # Put questions up the right way
        ax.invert_yaxis()
        # Show vertical grid bars
        ax.grid(True, axis="x")
        # Set vertical grid at 25% increments
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 0.25))
        # Convert fractional values (0.1) to percentages (10%)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        # Set the location of the legend (default=best)
        ax.legend(loc="best")
        # Add value text on top of bar
        for rec in ax.patches:
            # Determine the best text colour to use
            text_color = get_contrast_color(rec.get_facecolor())
            width = rec.get_width()
            if width == 0:
                continue
            ax.text(
                rec.get_x() + width / 2,
                rec.get_y() + rec.get_height() / 2,
                "{:.0f}%".format(width * 100),
                ha="center",
                va="center",
                color=text_color,
            )

        # Plot the comparisons
        ax2 = axes[i][1]
        if "comparison" in group:
            # Seperate negative and positive values so we can colour them differently
            negative_comparison = group.mask(group["comparison"].ge(0), other=np.nan)[["comparison"]]
            positive_comparison = group.mask(~group["comparison"].ge(0), other=np.nan)[["comparison"]]
            if not negative_comparison.empty:
                negative_comparison.plot(kind="barh", legend=False, align="center", width=0.2, ax=ax2, color=colors[3])
            if not positive_comparison.empty:
                positive_comparison.plot(kind="barh", legend=False, align="center", width=0.2, ax=ax2, color=colors[0])
        ax2.invert_yaxis()
        # Hide the grid and spine lines
        ax2.axis("off")
        # Center chart around 0
        ax2.set_xlim(-1, 1)
        # Display value
        for rec in ax2.patches:
            width = rec.get_width()
            if width == 0:
                continue
            text_color = colors[3] if width < 0 else colors[0]
            ax2.text(
                rec.get_x(),
                rec.get_y() - rec.get_height(),
                "{:.0f}".format(width * 100),
                ha="center",
                va="center",
                color=text_color,
            )
    return fig


@click.command()
@click.argument("input", type=click.File("rb"))
@click.argument("output")
def main(input, output):
    group2 = "What Team are you currently working on?"

    results = pd.read_csv(input)
    print(f"total respondents: {results.shape[0]}")

    results = results.rename(columns={group2: "team"}).set_index("team").select_dtypes(include="number")
    results = pd.get_dummies(results.stack()).reset_index(level=0)

    # Create dataframe of question <-> group
    question_groups = pd.DataFrame.from_dict(
        {
            "Team Dynamics": ["Team Charter", "Cross functional", "Collaboration", "Optimising flow"],
            "Vision and customer value": ["Vision and goals", "Success criteria", "Quality"],
            "Planning & Tracking": ["Iterative development", "Forecasting", "Data-driven planning"],
            "Continuous Delivery": ["Release cadence", "Confidence to release"],
            "How we build Software": [
                "Code quality",
                "Depth of testing",
                "Technical decisions",
                "Security",
                "Production health",
            ],
            "Continuous improvement and learning": ["Feedback", "Implemented improvement", "Team performance"],
            "Living Guru Values": [
                "Champion our customers",
                "Own our outcomes",
                "Learn all the things",
                "Be hungry, stay humble",
                "Keep it fun",
            ],
        },
        orient="index",
    )
    # Make question the index to match our data
    question_groups = (
        pd.DataFrame(question_groups.stack(), columns=["question"])
        .rename_axis(index=["domain", "id"])
        .reset_index(level=0)
        .set_index("question")
    )
    # Join to results
    results = (
        results.join(question_groups)
        .fillna({"domain": ""})
        .rename_axis("question")
        .set_index(["domain", "team"], append=True)
        .swaplevel(0, 2)
    )

    # Create aggregate results
    aggregate = results.groupby(level=[1, 2]).sum()
    # Convert counts into percentages
    aggregate = aggregate.div(aggregate.sum(axis=1), axis=0)
    # Calculate favourable score (sum of postitive responses)
    aggregate["favourable"] = aggregate.iloc[:, 0:2].sum(axis=1)

    # Count occurrences of responses
    results = results.groupby(level=[0, 1, 2]).sum()
    # Convert counts into percentages
    results = results.div(results.sum(axis=1), axis=0)
    # Calculate favourable score (sum of positive responses)
    results["favourable"] = results.iloc[:, 0:2].sum(axis=1)
    # Reverse order of columns
    # response_percent = response_percent.iloc[:, ::-1]

    results = results.reset_index(level=0).join(aggregate[["favourable"]], rsuffix="_all")
    results["comparison"] = results["favourable"] - results["favourable_all"]

    # Global figure styles
    plt.style.use("default")
    mpl.rc("axes", facecolor=facecolor)
    mpl.rc("figure", facecolor=facecolor)
    mpl.rc("axes.spines", left=False, bottom=False, top=False, right=False)
    mpl.rc("xtick.major", size=0)
    mpl.rc("ytick.major", size=0)

    aggregate_groups = aggregate.groupby(level="domain", dropna=False)

    fig = generate_graph(aggregate_groups, "All")
    print(f"writing to {output}...")
    fig.savefig(output, bbox_inches="tight")

    (root, ext) = os.path.splitext(output)
    by_team = results.groupby("team", dropna=False)
    for (key, group), i in zip(by_team, range(1, by_team.ngroups + 1)):
        alt_output = f"{root}.{i}{ext}"
        result_groups = group.groupby("domain", dropna=False)

        fig = generate_graph(result_groups, key)
        print(f"writing {i} of {result_groups.ngroups} to {alt_output}...")
        fig.savefig(alt_output, bbox_inches="tight")


if __name__ == "__main__":
    main()
