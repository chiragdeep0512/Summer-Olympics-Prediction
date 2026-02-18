def country_year_table(df):

    athletics = df.groupby(["NOC","Year"])['player_id'].nunique().reset_index()
    athletics.rename(columns={"player_id":"Total_Athletes"}, inplace=True)

    events = df.groupby(["NOC","Year"])['Event'].nunique().reset_index()
    events.rename(columns={"Event":"Total_Events"}, inplace=True)

    medal = df[df["Medal"] != "No Medal"]
    medal_counts = medal.groupby(["NOC","Year","Medal"]).size().unstack(fill_value=0).reset_index()

    final_df = athletics.merge(events, on=["NOC","Year"],how="left")
    final_df = final_df.merge(medal_counts, on=["NOC","Year"],how="left")

    for col in ["Gold","Silver","Bronze"]:
        if col not in final_df.columns:
            final_df[col] = 0

    final_df[["Gold", "Silver", "Bronze"]] = final_df[["Gold", "Silver", "Bronze"]].fillna(0)

    final_df["Total_Medals"] = (
        final_df["Gold"] + final_df["Silver"] + final_df["Bronze"]
    )

    final_df["Medal_Efficiency"] = (
        final_df["Total_Medals"] / final_df["Total_Athletes"]
    )

    return final_df
