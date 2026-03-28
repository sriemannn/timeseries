# /// script
# dependencies = [
#     "lightgbm>=4.6.0",
#     "marimo>=0.20.2",
#     "mcp==1.26.0",
#     "numpy==2.4.3",
#     "pandas==3.0.1",
#     "plotnine==0.15.3",
#     "pymc==5.28.2",
#     "pymc-extras>=0.10.0",
#     "pytest>=9.0.2",
#     "pyzmq>=27.1.0",
#     "scikit-misc>=0.5.2",
#     "skforecast>=0.19.1",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import pandas as pd
    from datetime import datetime, timedelta
    import arviz as az
    import numpy as np
    import plotnine as p9
    from mizani.breaks import breaks_date_width
    from mizani.labels import label_date

    from skforecast.recursive import ForecasterRecursiveMultiSeries
    from skforecast.preprocessing import RollingFeatures

    import matplotlib.style

    matplotlib.style.use(
        "default"
    )
    rng = np.random.default_rng(seed=123)


@app.cell
def _():
    def generate_sales_data(
        n_days: int = 7,
        n_lags: int = 3,
        n_categories: int = 9, 
        n_articles_per_category: int = 3, 
        rng: np.random.Generator | None = None
    ) -> pd.DataFrame: 

        if rng is None:
            rng = np.random.default_rng()

        idx_categories = np.arange(n_categories)

        mu_bar_categories = 150
        sigma_bar_categories = 40

        mu_categories = rng.normal(
            loc=mu_bar_categories, 
            scale=sigma_bar_categories,
            size=(n_categories, 1),
        )

        sigma_categories = rng.exponential(
            scale=40,
            size=(n_categories, 1),
        )

        stock = rng.normal(
            loc=mu_categories,
            scale=sigma_categories,
            size=(n_categories, n_articles_per_category),
        ).astype("int64")

        sales_rate = rng.poisson(
            7,
            size=(n_categories, n_articles_per_category)
        )

        start_date = datetime(2024, 1, 1)

        df_sales = pd.DataFrame(
            columns=["date", "category", "article", "sales", "stock"]
        )

        for day in range(n_days):
            date = start_date + timedelta(days=day)

            demand_fluctuation = rng.exponential(
                scale=0.7,
                size=(n_categories, n_articles_per_category)
            )

            demand = rng.poisson(
                (sales_rate * demand_fluctuation).astype("int64")
            )

            sales = np.min(
                np.stack((
                    demand,
                    stock
                )), 
                axis=0,
            )

            stock -= sales

            df = pd.DataFrame({
                "date": date,
                "category": np.repeat(idx_categories, n_articles_per_category),
                "article": np.tile(np.arange(n_articles_per_category), n_categories),
                "sales": sales.flatten(),
                "stock": stock.flatten(),
                "demand": demand.flatten(),
            })

            df_sales = pd.concat((df_sales, df), ignore_index=True)

        return (
            df_sales.assign(
                article=lambda df: df["category"] * n_articles_per_category + df["article"]
            ).astype({
                "date": "datetime64[ns]",
                "category": "category",
                "article": "category",
                "sales": "int64",
                "stock": "int64",
            })
            .sort_values(["article", "category", "date"])
            .reset_index(drop=True)
            .assign(
                **{
                    f"sales_t-{lag}": lambda df, lag=lag: (
                        df.groupby(["category", "article"])["sales"].shift(lag)
                    )
                    for lag in range(1, 1 + n_lags)
                }
            )
        )

    df_sales = generate_sales_data(
        n_days=31,
        n_articles_per_category=5,
        n_categories=12,
        rng=rng,
        n_lags=0
    )

    df_sales
    return (df_sales,)


@app.cell
def _(df_sales):
    (
        df_sales 
        .assign(
            _alpha=lambda df: df["stock"].case_when([
                (df["stock"] > 0, 1.0),
                (df["stock"] <= 0, 0.2),
            ])
        )
        >>
        p9.ggplot(
            mapping=p9.aes(
                x="date",
                y="stock",
                group="article",
                color="category",
                fill="category",
                alpha="_alpha",
            )
        )
        + p9.geom_line()
        + p9.geom_point(size=0.3)
        + p9.scale_x_datetime(
            breaks=breaks_date_width("7 day"),
            labels=label_date("%d")
        )
        + p9.facet_wrap(
            "category",
            labeller="label_both",
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=(7, 5),
            legend_position="none",
            axis_text_x=p9.element_text(
                rotation=90,
                vjust=0.5,
                hjust=1,
            )
        )
    )
    return


@app.cell
def _(df_sales):
    split_date = "2024-01-20"

    df_sales_date_index = df_sales.pivot(
        index="date", 
        columns="article",
        values="sales",
    ).rename(columns=lambda col: f"article_{col}").asfreq("D")


    df_sales_train = df_sales_date_index.query("date < @split_date")
    df_sales_test = df_sales_date_index.query("date >= @split_date")

    df_sales_train
    return (df_sales_train,)


@app.cell
def _(df_sales_train):
    from lightgbm import LGBMRegressor
    window_features = RollingFeatures(
        stats=["mean", "std", "min", "max"],
        window_sizes=7,
    )

    forecaster = ForecasterRecursiveMultiSeries(
        LGBMRegressor(verbose=-1),
        lags=5,
        window_features=window_features,
    )

    forecaster.fit(
        series=df_sales_train,
    )
    return (forecaster,)


@app.cell
def _(forecaster):
    forecaster.predict(
        steps=11,
    ).reset_index(names="date")
    return


@app.cell
def _(df_sales, forecaster):
    (
        df_sales
        .merge(
            forecaster.predict(steps=12).assign(
                article=lambda df: df["level"].str.replace("article_", "").astype("int64")
            ).reset_index(names="date"),
            on=["date", "article"],
            how="left"
        )
        .melt(
            id_vars=["date", "category", "article"],
            value_vars=["sales", "pred"],
            var_name="type",
        )
        .rename(columns={"value": "sales"})
        .query("article.isin([0, 1, 2])")
        >>
        p9.ggplot(
            mapping=p9.aes(
                x="date",
                y="sales",
                color="type",
                linetype="type",
            )
        )
        + p9.geom_line()
        + p9.facet_wrap(
            "article",
            labeller="label_both",
        )
        + p9.scale_x_datetime(
            breaks=breaks_date_width("7 day"),
            labels=label_date("%d")
        )
        + p9.scale_color_manual(
            values=["red", "black"],
            labels=["predicted", "actual"],
        )
        + p9.scale_linetype_manual(
            values=["dashed", "solid"],
            labels=["predicted", "actual"],
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=(7, 5),
            legend_position="top",
        )
    )
    return


if __name__ == "__main__":
    app.run()
