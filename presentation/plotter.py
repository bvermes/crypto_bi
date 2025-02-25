import plotly.graph_objects as go
import pandas as pd


def plot_fibonacci_chart(
    df, selected_index=None, title="Fibonacci Retracement", suffix=""
):
    """Plot a candlestick chart with Fibonacci levels using Plotly.

    Parameters:
    - df: DataFrame containing price and Fibonacci data.
    - selected_index: Index or list of indices to determine which Fibonacci levels to plot.
    - title: Title of the chart.
    - suffix: Suffix to match the correct Fibonacci columns.
    """

    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Candlesticks",
        )
    )

    # Adjust column names based on suffix
    swing_high_col = f"swing_high_val{suffix}"
    swing_low_col = f"swing_low_val{suffix}"

    # Add Swing Highs and Lows
    if swing_high_col in df.columns and swing_low_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df[df[swing_high_col].notna()]["timestamp"],
                y=df[df[swing_high_col].notna()][swing_high_col],
                mode="markers",
                marker=dict(color="green", size=8, symbol="triangle-up"),
                name="Swing Highs",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df[df[swing_low_col].notna()]["timestamp"],
                y=df[df[swing_low_col].notna()][swing_low_col],
                mode="markers",
                marker=dict(color="red", size=8, symbol="triangle-down"),
                name="Swing Lows",
            )
        )

    # Plot Fibonacci Levels using selected index
    fib_columns = [
        f"fib_23_6{suffix}",
        f"fib_38_2{suffix}",
        f"fib_50{suffix}",
        f"fib_61_8{suffix}",
        f"fib_78_6{suffix}",
        f"fib_100{suffix}",
        f"fib_161_8{suffix}",
        f"fib_261_8{suffix}",
    ]

    colors = ["cyan", "pink", "orange", "green", "purple", "blue", "yellow", "red"]

    # Allow user to select specific index to plot Fibonacci levels
    if selected_index is None:
        selected_index = df.index[-1]  # Default: Use last row
    elif isinstance(selected_index, list):
        selected_index = selected_index[-1]  # Take the last valid index from the list

    if selected_index in df.index:
        for level_name, color in zip(fib_columns, colors):
            if level_name in df.columns and not pd.isna(
                df.loc[selected_index, level_name]
            ):
                level_value = df.loc[selected_index, level_name]
                fig.add_trace(
                    go.Scatter(
                        x=[df["timestamp"].iloc[0], df["timestamp"].iloc[-1]],
                        y=[level_value, level_value],
                        mode="lines",
                        line=dict(color=color, width=1, dash="dash"),
                        name=f"{level_name}: {level_value:.2f}",
                    )
                )

    # Chart layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price",
        legend_title="Legend",
        template="plotly_dark",
        height=600,
        width=1000,
    )

    fig.update_xaxes(rangeslider_visible=False, type="date")
    fig.update_yaxes(tickprefix="$")

    fig.show()
