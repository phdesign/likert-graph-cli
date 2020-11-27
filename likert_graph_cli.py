import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
from matplotlib.colors import to_rgba

@click.command()
@click.argument('input', type=click.File('rb'))
@click.argument('output', type=click.File('wb'))
def main(input, output):
    group2 = 'What Team are you currently working on?'

    results = pd.read_csv(input)
    print(f'Total respondents: {results.shape[0]}')

    multichoice = results.select_dtypes(include='number').append(results[[group2]])

    # Create aggregate results
    aggregate = multichoice.drop(columns=[group2])
    # Count occurrences of responses
    aggregate_response_count = pd.get_dummies(aggregate.stack()).groupby(level=[1]).sum()
    # Give index (questions) a name
    aggregate_response_count.rename_axis('question', inplace=True)
    # Convert counts into percentages
    aggregate_percent = aggregate_response_count.div(aggregate_response_count.sum(axis=1), axis=0)
    # Calculate favourable score (sum of postitive responses)
    aggregate_percent['favourable'] = aggregate_percent.iloc[:, 0:2].sum(axis=1)

    # Add team to multi index
    team_results = multichoice.set_index(group2, append=True)
    # Count occurrences of responses
    response_count = pd.get_dummies(team_results.stack()).groupby(level=[1,2]).sum()
    # Give index (questions) a name
    #response_count.index.rename('question', level=0)
    response_count.rename_axis(index=['team', 'question'], inplace=True)
    # Convert counts into percentages
    response_percent = response_count.div(response_count.sum(axis=1), axis=0)
    # Calculate favourable score (sum of positive responses)
    response_percent['favourable'] = response_percent.iloc[:, 0:2].sum(axis=1)
    # Reverse order of columns
    #response_percent = response_percent.iloc[:, ::-1]

    # Create dataframe of question <-> group
    question_groups = pd.DataFrame.from_dict({
        'Team Dynamics': ['Team Charter', 'Cross functional', 'Collaboration', 'Optimising flow'],
        'Vision and customer value': ['Vision and goals','Success criteria', 'Quality'],
        'Planning & Tracking': ['Iterative development','Forecasting', 'Data-driven planning'],
        'Continuous Delivery': ['Release cadence', 'Confidence to release'],
        'How we build Software': ['Code quality', 'Depth of testing', 'Technical decisions', 'Security', 'Production health'],
        'Continuous improvement and learning': ['Feedback', 'Implemented improvement', 'Team performance'],
        'Living Guru Values': ['Champion our customers', 'Own our outcomes', 'Learn all the things', 'Be hungry, stay humble', 'Keep it fun']
    }, orient='index')
    # Make question the index to match our data
    question_groups = pd.DataFrame(question_groups.stack(), columns=['Question'])\
        .rename_axis(index=['Domain', 'Id'])\
        .reset_index().drop(columns=['Id'])\
        .set_index('Question')

    grouped_data = aggregate_percent\
        .join(question_groups)\
        .fillna({'Domain': ''})\
        .groupby('Domain', dropna=False)

    # Global figure styles
    colors = ['#466384', '#869caf', '#e4af8e', '#d67242']
    facecolor = '#f2f5fc'

    plt.style.use('default')
    mpl.rc('axes', facecolor=facecolor)
    mpl.rc('figure', facecolor=facecolor)
    mpl.rc('axes.spines', left=False, bottom=False, top=False, right=False)
    mpl.rc('xtick.major', size=0)
    mpl.rc('ytick.major', size=0)

    def get_contrast_color(facecolor):
        """Calculate the contrasting text colour.

        Uses euclidean distance between the facecolor and text colours
        to determine if black or white is more contrasting.
        """
        facecolor = np.array(facecolor)
        white = np.array(to_rgba('white'))
        black = np.array(to_rgba('black'))
        white_dist = np.linalg.norm(facecolor-white)
        black_dist = np.linalg.norm(facecolor-black)
        return white if white_dist > black_dist else black

    # Calculate the number of subplots based on our groups
    ncols = 2
    nrows = grouped_data.ngroups
    fig = plt.figure(figsize=(10,30))
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        hspace=0.5, # Vertical space between subplots
        wspace=0,
        height_ratios=grouped_data.size().tolist(),
        width_ratios=([0.8, 0.2] if 'comparison' in grouped_data.count() else [1, 0])
    )
    axes = gs.subplots(sharey='row')

    # Draw subplot for each group
    for (key, group), i in zip(grouped_data, range(grouped_data.ngroups)):
        ax = group[[1, 2, 3, 4]].plot(
            kind='barh',
            stacked=True,
            color=colors,
            legend=True,
            xlabel='',
            fontsize=10,
            ax=axes[i][0],
            title=key,
            edgecolor=facecolor
        )
        # Add padding between question & graph
        ax.tick_params(axis='y', which='major', pad=30)
        # Put questions up the right way
        ax.invert_yaxis()
        # Show vertical grid bars
        ax.grid(True, axis='x')
        # Set vertical grid at 25% increments
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.arange(start, end, 0.25))
        # Convert fractional values (0.1) to percentages (10%)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        # Set the location of the legend (default=best)
        ax.legend(loc='best')
        # Add value text on top of bar
        for rec in ax.patches:
            # Determine the best text colour to use
            text_color = get_contrast_color(rec.get_facecolor())
            width = rec.get_width()
            if width == 0:
                continue
            ax.text(rec.get_x() + width / 2,
                    rec.get_y() + rec.get_height() / 2,
                    "{:.0f}%".format(width * 100),
                    ha='center',
                    va='center',
                    color=text_color)

        # Plot the comparisons
        ax2 = axes[i][1]
        if 'comparison' in group:
            # Seperate negative and positive values so we can colour them differently
            negative_comparison = group.mask(group['comparison'].ge(0), other=np.nan)[['comparison']]
            positive_comparison = group.mask(~group['comparison'].ge(0), other=np.nan)[['comparison']]
            if not negative_comparison.empty:
                negative_comparison.plot(
                    kind='barh',
                    legend=False,
                    align='center',
                    width=0.2,
                    ax=ax2,
                    color=colors[3]
                )
            if not positive_comparison.empty:
                positive_comparison.plot(
                    kind='barh',
                    legend=False,
                    align='center',
                    width=0.2,
                    ax=ax2,
                    color=colors[0]
                )
        ax2.invert_yaxis()
        # Hide the grid and spine lines
        ax2.axis('off')
        # Center chart around 0
        ax2.set_xlim(-1, 1)
        # Display value
        for rec in ax2.patches:
            width = rec.get_width()
            if width == 0:
                continue
            text_color = colors[3] if width < 0 else colors[0]
            ax2.text(rec.get_x(),
                    rec.get_y() - rec.get_height(),
                    "{:.0f}".format(width * 100),
                    ha='center',
                    va='center',
                    color=text_color)

    plt.savefig(output, bbox_inches='tight')

if __name__ == "__main__":
    main()