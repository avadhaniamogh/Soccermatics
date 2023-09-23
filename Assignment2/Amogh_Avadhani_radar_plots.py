import pandas as pd
import numpy as np
import json
# plotting
import os
import pathlib
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from joblib import load
from sklearn.linear_model import LinearRegression
from scipy import stats
from mplsoccer import PyPizza, FontManager
import time

start_time = time.time()

pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')


def get_events(file_name):
    df = pd.DataFrame()
    path = os.path.join(str(pathlib.Path().resolve()), 'events', file_name)
    with open(path) as f:
        data = json.load(f)
    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)
    df = df.reset_index()
    return df


# npxG
def calulate_npxG(dataframe):
    # very basic xG model based on
    shots = dataframe.loc[dataframe["eventName"] == "Shot"].copy()
    shots["X"] = shots.positions.apply(lambda cell: (100 - cell[0]['x']) * 105 / 100)
    shots["Y"] = shots.positions.apply(lambda cell: cell[0]['y'] * 68 / 100)
    shots["C"] = shots.positions.apply(lambda cell: abs(cell[0]['y'] - 50) * 68 / 100)
    # calculate distance and angle
    shots["Distance"] = np.sqrt(shots["X"] ** 2 + shots["C"] ** 2)
    shots["Angle"] = np.where(np.arctan(7.32 * shots["X"] / (shots["X"] ** 2 + shots["C"] ** 2 - (7.32 / 2) ** 2)) > 0,
                              np.arctan(7.32 * shots["X"] / (shots["X"] ** 2 + shots["C"] ** 2 - (7.32 / 2) ** 2)),
                              np.arctan(
                                  7.32 * shots["X"] / (shots["X"] ** 2 + shots["C"] ** 2 - (7.32 / 2) ** 2)) + np.pi)
    # if you ever encounter problems (like you have seen that model treats 0 as 1 and 1 as 0) while modelling -
    # change the dependant variable to object
    shots["Goal"] = shots.tags.apply(lambda x: 1 if {'id': 101} in x else 0).astype(object)
    # headers have id = 403
    headers = shots.loc[shots.apply(lambda x: {'id': 403} in x.tags, axis=1)]
    non_headers = shots.drop(headers.index)

    headers_model = smf.glm(formula="Goal ~ Distance + Angle", data=headers,
                            family=sm.families.Binomial()).fit()
    # non-headers
    nonheaders_model = smf.glm(formula="Goal ~ Distance + Angle", data=non_headers,
                               family=sm.families.Binomial()).fit()
    # assigning xG
    # headers
    b_head = headers_model.params
    xG = 1 / (1 + np.exp(b_head[0] + b_head[1] * headers['Distance'] + b_head[2] * headers['Angle']))
    headers = headers.assign(xG=xG)

    # non-headers
    b_nhead = nonheaders_model.params
    xG = 1 / (1 + np.exp(b_nhead[0] + b_nhead[1] * non_headers['Distance'] + b_nhead[2] * non_headers['Angle']))
    non_headers = non_headers.assign(xG=xG)

    # concat, group and sum
    all_shots_xg = pd.concat([non_headers[["playerId", "xG"]], headers[["playerId", "xG"]]])
    all_shots_xg.rename(columns={"xG": "npxG"}, inplace=True)
    xG_sum = all_shots_xg.groupby(["playerId"])["npxG"].sum().sort_values(ascending=False).reset_index()
    # group by player and sum

    return xG_sum


# npxg = calulatenpxG(df)

# Smart passes
def get_smart_passes(dataframe):
    # get smart passes
    smart_passes = dataframe.loc[dataframe["subEventName"] == "Smart pass"]
    # find accurate
    smart_passes_made = smart_passes.loc[smart_passes.apply(lambda x: {'id': 1801} in x.tags, axis=1)]

    # sum by player
    sp_player = smart_passes_made.groupby(["playerId"]).eventId.count().reset_index()
    sp_player.rename(columns={'eventId': 'smart_passes'}, inplace=True)

    return sp_player


# smart_passes = smartPasses(df)
# print(smart_passes.head(3))

# TODO: change this
def assists_key_passes(dataframe):
    # get assists
    passes = dataframe.loc[dataframe["eventName"] == "Pass"]
    assists = passes.loc[passes.apply(lambda x: {'id': 301} in x.tags, axis=1)]
    # get key passes
    key_passes = passes.loc[passes.apply(lambda x: {'id': 302} in x.tags, axis=1)]

    # assists by player
    a_player = assists.groupby(["playerId"]).eventId.count().reset_index()
    a_player.rename(columns={'eventId': 'assists'}, inplace=True)

    # key passes by player
    kp_player = key_passes.groupby(["playerId"]).eventId.count().reset_index()
    kp_player.rename(columns={'eventId': 'key_passes'}, inplace=True)

    player_data = a_player.merge(kp_player, how="outer", on=["playerId"])
    return player_data


# gakp = GoalsAssistsKeyPasses(df)
# gakp.head(3)

def get_minutes_per_game(file_name):
    path = os.path.join(str(pathlib.Path().resolve()), file_name)
    with open(path) as f:
        minutes_per_game = json.load(f)
    minutes_per_game = pd.DataFrame(minutes_per_game)
    return minutes_per_game


def get_players():
    path = os.path.join(str(pathlib.Path().resolve()), 'players.json')
    with open(path) as f:
        players = json.load(f)
    player_df = pd.DataFrame(players)
    return player_df


def get_players_role(player_df, role):
    player_subset = player_df.loc[player_df.apply(lambda x: x.role["name"] == role, axis=1)]
    player_subset.rename(columns={'wyId': 'playerId'}, inplace=True)
    to_return = player_subset[['playerId', 'shortName']]
    return to_return


def get_possession_percentage_df(minutes_per_game):
    possesion_dict = {}
    # for every row in the dataframe
    for i, row in minutes_per_game.iterrows():
        # take player id, team id and match id, minute in and minute out
        player_id, team_id, match_id = row["playerId"], row["teamId"], row["matchId"]
        # create a key in dictionary if player encounterd first time
        if not str(player_id) in possesion_dict.keys():
            possesion_dict[str(player_id)] = {'team_passes': 0, 'all_passes': 0}
        min_in = row["player_in_min"] * 60
        min_out = row["player_out_min"] * 60

        # get the dataframe of events from the game
        match_df = df.loc[df["matchId"] == match_id].copy()
        # add to 2H the highest value of 1H
        match_df.loc[match_df["matchPeriod"] == "2H", 'eventSec'] = match_df.loc[
                                                                        match_df["matchPeriod"] == "2H", 'eventSec'] + \
                                                                    match_df.loc[match_df["matchPeriod"] == "1H"][
                                                                        "eventSec"].iloc[-1]
        # take all events from this game and this period
        player_in_match_df = match_df.loc[match_df["eventSec"] > min_in].loc[match_df["eventSec"] <= min_out]
        # take all passes and won duels as described
        all_passes = player_in_match_df.loc[player_in_match_df["eventName"].isin(["Pass", "Duel"])]
        # adjusting for no passes in this period (Tuanzebe)
        if len(all_passes) > 0:
            # removing lost air duels
            no_contact = all_passes.loc[
                all_passes["subEventName"].isin(["Air duel", "Ground defending duel", "Ground loose ball duel"])].loc[
                all_passes.apply(lambda x: {'id': 701} in x.tags, axis=1)]
            all_passes = all_passes.drop(no_contact.index)
        # take team passes
        team_passes = all_passes.loc[all_passes["teamId"] == team_id]
        # append it {player id: {team passes: sum, all passes : sum}}
        possesion_dict[str(player_id)]["team_passes"] += len(team_passes)
        possesion_dict[str(player_id)]["all_passes"] += len(all_passes)

    # calculate possesion for each player
    percentage_dict = {key: value["team_passes"] / value["all_passes"] if value["all_passes"] > 0 else 0 for key, value
                       in
                       possesion_dict.items()}
    # create a dataframe
    percentage_df = pd.DataFrame(percentage_dict.items(), columns=["playerId", "possesion"])
    percentage_df["playerId"] = percentage_df["playerId"].astype(int)
    return percentage_df


def draw_radar(summary_df_adjusted, player_df_adjusted, short_name, league_name):
    adjusted_columns = player_df_adjusted.columns[:]
    # values
    values = [player_df_adjusted[column].iloc[0] for column in adjusted_columns]
    # percentiles
    percentiles = [int(stats.percentileofscore(summary_df_adjusted[column], player_df_adjusted[column].iloc[0])) for
                   column in
                   adjusted_columns]
    names = ["non-penalty Expected Goals", "Assists", "Key Passes", "Smart Passes"]

    slice_colors = ["blue"] * 1 + ["green"] * 2 + ["red"] * 1
    text_colors = ["white"] * 4
    font_normal = FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/"
                               "Roboto%5Bwdth,wght%5D.ttf?raw=true"))
    font_italic = FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/"
                               "Roboto-Italic%5Bwdth,wght%5D.ttf?raw=true"))
    font_bold = FontManager(("https://github.com/google/fonts/blob/main/apache/robotoslab/"
                             "RobotoSlab%5Bwght%5D.ttf?raw=true"))

    baker = PyPizza(
        params=names,  # list of parameters
        straight_line_color="#000000",  # color for straight lines
        straight_line_lw=1,  # linewidth for straight lines
        last_circle_lw=1,  # linewidth of last circle
        other_circle_lw=1,  # linewidth for other circles
        other_circle_ls="-."  # linestyle for other circles
    )

    fig, ax = baker.make_pizza(
        percentiles,  # list of values
        figsize=(10, 10),  # adjust figsize according to your need
        param_location=110,
        slice_colors=slice_colors,
        value_colors=text_colors,
        value_bck_colors=slice_colors,
        # where the parameters will be added
        kwargs_slices=dict(
            facecolor="cornflowerblue", edgecolor="#000000",
            zorder=2, linewidth=1
        ),  # values to be used when plotting slices
        kwargs_params=dict(
            color="#000000", fontsize=12,
            fontproperties=font_normal.prop, va="center"
        ),  # values to be used when adding parameter
        kwargs_values=dict(
            color="#000000", fontsize=12,
            fontproperties=font_normal.prop, zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor="cornflowerblue",
                boxstyle="round,pad=0.2", lw=1
            )
        )  # values to be used when adding parameter-values
    )

    # add title
    fig.text(
        0.515, 0.97, short_name + " per 90 (possesion adjusted)", size=18,
        ha="center", fontproperties=font_bold.prop, color="#000000"
    )

    # add subtitle
    fig.text(
        0.515, 0.942,
        "Percentile Rank vs " + league_name + " Midfielders | Season 2017-18",
        size=15,
        ha="center", fontproperties=font_bold.prop, color="#000000"
    )

    plt.show()


# TODO : make it to return a df and draw radar later
def process_radar(df, short_name, minutes, player_df_role, percentage_df):
    npxg = calulate_npxG(df)
    smart_passes = get_smart_passes(df)
    akp = assists_key_passes(df)
    players = df["playerId"].unique()
    summary = pd.DataFrame(players, columns=["playerId"])
    summary = summary.merge(npxg, how="left", on=["playerId"]) \
        .merge(smart_passes, how="left", on=["playerId"]) \
        .merge(akp, how="left", on=["playerId"])
    summary = summary.merge(minutes, how="left", on=["playerId"])
    summary = summary.fillna(0)
    summary = summary.loc[summary["minutesPlayed"] > 400]
    # Merge with players.json role subset
    summary = summary.merge(player_df_role, how="inner", on=["playerId"])

    summary = summary.merge(percentage_df, how="left", on=["playerId"])
    summary_adjusted = pd.DataFrame()
    summary_adjusted["shortName"] = summary["shortName"]
    # calculate value adjusted
    for column in summary.columns[1:5]:
        summary_adjusted[column + "_adjusted_per90"] = summary.apply(
            lambda x: (x[column] / x["possesion"]) * 90 / x["minutesPlayed"], axis=1)

    return summary_adjusted


player_df = get_players()
'''
English league
'''
# league_name = "English Premier League"
# events_file_name = 'events_England.json'
# df = get_events(events_file_name)
# minutes_file_name = "minutes_played_per_game_England.json"
# minutes_per_game = get_minutes_per_game(minutes_file_name)
# minutes = minutes_per_game.groupby(["playerId"]).minutesPlayed.sum().reset_index()
# percentage_df = get_possession_percentage_df(minutes_per_game)
'''
Bournemouth players
'''
# shortName = "M. Pugh"
# role = "Midfielder"
# player_role_df = get_players_role(player_df, role)
# summary_adjusted = process_radar(df, shortName, minutes, player_role_df, percentage_df)
# player_adjusted = summary_adjusted.loc[summary_adjusted["shortName"] == short_name]
# player_adjusted = player_adjusted[['npxG_adjusted_per90', "assists_adjusted_per90", "key_passes_adjusted_per90",
#                                        "smart_passes_adjusted_per90"]]
# draw_radar(summary_adjusted, player_adjusted, shortName, league_name)

# shortName = "M. Pugh"
# role = "Midfielder"
# player_role_df = get_players_role(player_df, role)
# summary_adjusted = process_radar(df, shortName, minutes, player_role_df, percentage_df)
# player_adjusted = summary_adjusted.loc[summary_adjusted["shortName"] == short_name]
# player_adjusted = player_adjusted[['npxG_adjusted_per90', "assists_adjusted_per90", "key_passes_adjusted_per90",
#                                        "smart_passes_adjusted_per90"]]
# draw_radar(summary_adjusted, player_adjusted, shortName, league_name)

'''
Italian league
'''
league_name = "Serie A"
events_file_name = 'events_Italy.json'
df = get_events(events_file_name)
minutes_file_name = "minutes_played_per_game_Italy.json"
minutes_per_game = get_minutes_per_game(minutes_file_name)
minutes = minutes_per_game.groupby(["playerId"]).minutesPlayed.sum().reset_index()
percentage_df = get_possession_percentage_df(minutes_per_game)
'''
Recommended player
'''
shortName = "R. de Paul"
role = "Midfielder"
player_role_df = get_players_role(player_df, role)
summary_adjusted = process_radar(df, shortName, minutes, player_role_df, percentage_df)
player_adjusted = summary_adjusted.loc[summary_adjusted["shortName"] == shortName]
player_adjusted = player_adjusted[['npxG_adjusted_per90', "assists_adjusted_per90", "key_passes_adjusted_per90",
                                   "smart_passes_adjusted_per90"]]
draw_radar(summary_adjusted, player_adjusted, shortName, league_name)

# path = os.path.join(str(pathlib.Path().resolve()), 'minutes_played_per_game_England.json')
# with open(path) as f:
#     minutes_per_game = json.load(f)
# minutes_per_game = pd.DataFrame(minutes_per_game)
# minutes = minutes_per_game.groupby(["playerId"]).minutesPlayed.sum().reset_index()
#
# players = df["playerId"].unique()
# summary = pd.DataFrame(players, columns=["playerId"])
# summary = summary.merge(npxg, how="left", on=["playerId"])\
#                 .merge(smart_passes, how="left", on=["playerId"])\
#                 .merge(gakp, how="left", on=["playerId"])
#
# summary = minutes.merge(summary, how="left", on=["playerId"])
# summary = summary.fillna(0)
# summary = summary.loc[summary["minutesPlayed"] > 400]

# print(summary.head(5))

# path = os.path.join(str(pathlib.Path().resolve()), 'players.json')
# with open(path) as f:
#     players = json.load(f)
# player_df = pd.DataFrame(players)
# forwards = player_df.loc[player_df.apply(lambda x: x.role["name"] == "Midfielder", axis=1)]
# forwards.rename(columns={'wyId': 'playerId'}, inplace=True)
# to_merge = forwards[['playerId', 'shortName']]
# summary = summary.merge(to_merge, how="inner", on=["playerId"])

# possesion_dict = {}
# # for every row in the dataframe
# for i, row in minutes_per_game.iterrows():
#     # take player id, team id and match id, minute in and minute out
#     player_id, team_id, match_id = row["playerId"], row["teamId"], row["matchId"]
#     # create a key in dictionary if player encounterd first time
#     if not str(player_id) in possesion_dict.keys():
#         possesion_dict[str(player_id)] = {'team_passes': 0, 'all_passes': 0}
#     min_in = row["player_in_min"] * 60
#     min_out = row["player_out_min"] * 60
#
#     # get the dataframe of events from the game
#     match_df = df.loc[df["matchId"] == match_id].copy()
#     # add to 2H the highest value of 1H
#     match_df.loc[match_df["matchPeriod"] == "2H", 'eventSec'] = match_df.loc[
#                                                                     match_df["matchPeriod"] == "2H", 'eventSec'] + \
#                                                                 match_df.loc[match_df["matchPeriod"] == "1H"][
#                                                                     "eventSec"].iloc[-1]
#     # take all events from this game and this period
#     player_in_match_df = match_df.loc[match_df["eventSec"] > min_in].loc[match_df["eventSec"] <= min_out]
#     # take all passes and won duels as described
#     all_passes = player_in_match_df.loc[player_in_match_df["eventName"].isin(["Pass", "Duel"])]
#     # adjusting for no passes in this period (Tuanzebe)
#     if len(all_passes) > 0:
#         # removing lost air duels
#         no_contact = all_passes.loc[
#             all_passes["subEventName"].isin(["Air duel", "Ground defending duel", "Ground loose ball duel"])].loc[
#             all_passes.apply(lambda x: {'id': 701} in x.tags, axis=1)]
#         all_passes = all_passes.drop(no_contact.index)
#     # take team passes
#     team_passes = all_passes.loc[all_passes["teamId"] == team_id]
#     # append it {player id: {team passes: sum, all passes : sum}}
#     possesion_dict[str(player_id)]["team_passes"] += len(team_passes)
#     possesion_dict[str(player_id)]["all_passes"] += len(all_passes)
#
# # calculate possesion for each player
# percentage_dict = {key: value["team_passes"] / value["all_passes"] if value["all_passes"] > 0 else 0 for key, value in
#                    possesion_dict.items()}
# # create a dataframe
# percentage_df = pd.DataFrame(percentage_dict.items(), columns=["playerId", "possesion"])
# percentage_df["playerId"] = percentage_df["playerId"].astype(int)
# print(percentage_df.head(20))
# merge it
# summary = summary.merge(percentage_df, how="left", on=["playerId"])
# print(summary.head(5))

# create a new dataframe only for it
# summary_adjusted = pd.DataFrame()
# summary_adjusted["shortName"] = summary["shortName"]
# # calculate value adjusted
# for column in summary.columns[2:6]:
#     summary_adjusted[column + "_adjusted_per90"] = summary.apply(
#         lambda x: (x[column] / x["possesion"]) * 90 / x["minutesPlayed"], axis=1)
#
# print(summary_adjusted.head(50))

'''
Bournemouth FC Player #1 - Marc Pugh
'''
# short_name = "M. Pugh"
# # only his statistics
# stanislas_adjusted = summary_adjusted.loc[summary_adjusted["shortName"] == short_name]
# pd.set_option('display.max_columns', None)
# # print(stanislas_adjusted.head(5))
# stanislas_adjusted = stanislas_adjusted[['npxG_adjusted_per90', "assists_adjusted_per90", "key_passes_adjusted_per90",
#                                          "smart_passes_adjusted_per90"]]
# # take only necessary columns
# adjusted_columns = stanislas_adjusted.columns[:]
# # values
# values = [stanislas_adjusted[column].iloc[0] for column in adjusted_columns]
# # percentiles
# percentiles = [int(stats.percentileofscore(summary_adjusted[column], stanislas_adjusted[column].iloc[0])) for column in
#                adjusted_columns]
# names = names = ["non-penalty Expected Goals", "Smart Passes", "Assists", "Key Passes"]
#
# slice_colors = ["blue"] * 1 + ["green"] * 2 + ["red"] * 1
# text_colors = ["white"] * 4
# font_normal = FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/"
#                            "Roboto%5Bwdth,wght%5D.ttf?raw=true"))
# font_italic = FontManager(("https://github.com/google/fonts/blob/main/apache/roboto/"
#                            "Roboto-Italic%5Bwdth,wght%5D.ttf?raw=true"))
# font_bold = FontManager(("https://github.com/google/fonts/blob/main/apache/robotoslab/"
#                          "RobotoSlab%5Bwght%5D.ttf?raw=true"))
#
# baker = PyPizza(
#     params=names,  # list of parameters
#     straight_line_color="#000000",  # color for straight lines
#     straight_line_lw=1,  # linewidth for straight lines
#     last_circle_lw=1,  # linewidth of last circle
#     other_circle_lw=1,  # linewidth for other circles
#     other_circle_ls="-."  # linestyle for other circles
# )
#
# fig, ax = baker.make_pizza(
#     percentiles,  # list of values
#     figsize=(10, 10),  # adjust figsize according to your need
#     param_location=110,
#     slice_colors=slice_colors,
#     value_colors=text_colors,
#     value_bck_colors=slice_colors,
#     # where the parameters will be added
#     kwargs_slices=dict(
#         facecolor="cornflowerblue", edgecolor="#000000",
#         zorder=2, linewidth=1
#     ),  # values to be used when plotting slices
#     kwargs_params=dict(
#         color="#000000", fontsize=12,
#         fontproperties=font_normal.prop, va="center"
#     ),  # values to be used when adding parameter
#     kwargs_values=dict(
#         color="#000000", fontsize=12,
#         fontproperties=font_normal.prop, zorder=3,
#         bbox=dict(
#             edgecolor="#000000", facecolor="cornflowerblue",
#             boxstyle="round,pad=0.2", lw=1
#         )
#     )  # values to be used when adding parameter-values
# )
#
# # add title
# fig.text(
#     0.515, 0.97, short_name + " per 90 (possesion adjusted) - Bournemouth FC", size=18,
#     ha="center", fontproperties=font_bold.prop, color="#000000"
# )
#
# # add subtitle
# fig.text(
#     0.515, 0.942,
#     "Percentile Rank vs Premier League Midfielders | Season 2017-18",
#     size=15,
#     ha="center", fontproperties=font_bold.prop, color="#000000"
# )
#
# plt.show()


print(time.time() - start_time, "seconds - Code execution time")
