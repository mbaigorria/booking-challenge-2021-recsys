import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


def get_plot_from_accuracy(**kwargs) -> None:
    """
    Accuracy plot by position.
    """
    df_list = []
    for key, accuracy_dict in kwargs.items():
        df = pd.DataFrame.from_dict(accuracy_dict, orient='index', columns=['accuracy']).head(8)
        df['type'] = key
        df_list.append(df)
    g = sns.catplot(
        data=pd.concat(df_list).reset_index(), kind="bar",
        x="index", y="accuracy", hue="type",
        palette="bone", height=6, legend_out=False
    )
    g.set(ylim=(0.4, 0.7))
    g.set_axis_labels("Sequence length", "accuracy@4")
    g.savefig("accuracy_by_position.pdf")


def get_plot_from_distribution_by_pos(df: pd.DataFrame):
    """
    Plot distribution by position from dataframe.
    """
    df_melt = pd.melt(df,
                      value_vars=['train_set', 'submission'],
                      var_name='dataset_type',
                      value_name='sequence_length',
                      ignore_index=False)

    sns.set_style('white')
    sns.set_context('paper', font_scale=2)
    sns.set_palette(['#000000', '#ABABAB'])
    sns.set_style('ticks', {'axes.edgecolor': '0',
                            'xtick.color': '0',
                            'ytick.color': '0'})

    g = sns.catplot(
        data=df_melt.reset_index(), kind="bar",
        x="index", y="sequence_length", hue="dataset_type",
        ci="sd", height=6, legend_out=False,
    )
    g.set_axis_labels("Sequence length", "Proportion")
    new_labels = ['Training set', 'Submission set']
    for t, l in zip(g._legend.texts, new_labels):
        t.set_text(l)

    g._legend.set_title('')
    g.savefig("sequence_length_distribution.pdf")
