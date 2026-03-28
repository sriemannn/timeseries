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
    )

    df_sales
    return (df_sales,)


@app.cell
def _(df_sales):
    df_sales_grouped = (
        df_sales
        .sort_values("date")
        .groupby(["category", "article"])
    )

    split_size = 0.8

    df_sales_train = (
        df_sales_grouped
        .apply(lambda df: df.iloc[:int(len(df) * split_size)])
        .reset_index()
    )

    df_sales_test = (
        df_sales_grouped
        .apply(lambda df: df.iloc[int(len(df) * split_size):])
        .reset_index()
    )

    X_train, y_train = (
        df_sales_train
        .filter(regex="^sales_t-"),
        df_sales_train["sales"]
    )

    X_test, y_test = (
        df_sales_test
        .filter(regex="^sales_t-"),
        df_sales_test["sales"]
    )
    return X_test, X_train, df_sales_test, df_sales_train, y_train


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
def _(X_train, y_train):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=100,
        random_state=123,
    )

    model.fit(X_train, y_train)

    return (model,)


@app.cell
def _(X_test, df_sales_test, model):
    df_sales_test_pred = (
        df_sales_test
        .assign(
            sales_pred=model.predict(X_test),
        )
    )

    df_sales_test_pred
    return (df_sales_test_pred,)


@app.cell
def _(df_sales_test_pred, df_sales_train):
    df_plot_pred = pd.concat([
        df_sales_train.assign(
            sales_pred=None
        ), 
        df_sales_test_pred
    ])
    return (df_plot_pred,)


@app.cell
def _(df_sales_test):
    df_sales_test
    return


@app.cell
def _(df_plot_pred):
    random_articles = rng.choice(
        df_plot_pred["article"].unique(),
        size=3,
        replace=False,
    )

    (
        df_plot_pred
        .query("article.isin(@random_articles)")
        .melt(
            id_vars=["date", "category", "article"],
            value_vars=["sales", "sales_pred"],
            var_name="type",
            value_name="n_sales",
        )
        .rename(columns={"n_sales": "sales"})
        >>
        p9.ggplot(
            mapping=p9.aes(
                x="date",
                y="sales",
                group="type",
                color="type",
                fill="type",
                linetype="type"
            )
        )
        + p9.geom_line()
        + p9.scale_x_datetime(
            breaks=breaks_date_width("7 day"),
            labels=label_date("%d")
        )
        + p9.facet_wrap(
            "article",
            labeller="label_both",
        )
        + p9.theme_minimal()
        + p9.theme(
            figure_size=(7, 3),
            legend_position="bottom",
            axis_text_x=p9.element_text(
                rotation=90,
                vjust=0.5,
                hjust=1,
            )
        )
        + p9.coord_cartesian(ylim=(0, None))
    )
    return


if __name__ == "__main__":
    app.run()
