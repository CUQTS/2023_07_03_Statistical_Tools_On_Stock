# Implied Volatility Project

## Description

This project aims to analyze and predict financial market volatility and price movement using Python programming and statistical methodologies. The goal is to understand the complexities of volatility, explore its risks, and identify potential opportunities for traders and investors. The research serves as a foundation for refining investment strategies, risk management approaches, and financial forecasting processes.

One notable aspect of this project is its interactive nature. Users have the opportunity to actively participate by providing input on various parameters. They can enter the ticker symbol for their asset of interest, specify the look-back period for the daily return distribution, indicate the target confidence level for the confidence interval of the next-day close price, set the moneyness boundary for the implied volatility surface, and determine the length of the rolling window for Z-score calculations. By exploring different models and adjusting these parameters, we aim to provide comprehensive insights into market volatility. This knowledge can serve as a valuable guide for making informed investment and trading decisions, as well as developing robust risk management strategies.

## Features

1. Model Selection (GARCH and EWMA)
2. Volatility Predictions
3. Implied Volatility Surface
4. Daily Return Distribution
5. Confidence Interval on Next Day Close Price
6. Rolling Z-Score on Close Price

## Getting Started

To run the project, follow these steps:

Open the terminal/command prompt.

Navigate to the project directory.

Run the following command:

`python3 CUQTSarticle_with_IV.py`

## Usage

### Model Selection

After running the script, you will be prompted to enter the ticker symbol according to Yahoo Finance. For example, if you want to analyze AAPL (Apple Inc.), enter "AAPL".

The script will calculate the daily, monthly, and annual volatility of the specified stock (AAPL in this case) using the selected GARCH model.

The selected GARCH model and the chosen EWMA lambda will be displayed on the screen.

A graph showing the previous volatility and implied volatility will be displayed.

### Implied Volatility Surface

Next, you will be prompted to enter the strike level bound, which should be a multiple of 5 and in the range of 5 to 95.

The script will plot the implied volatility surface for call and put options.

### Daily Return Distribution

You will then be prompted to enter the length of the look-back period for the daily return distribution.

The script will plot the daily return distribution for the specified look-back period.

The mean, volatility, skewness, and kurtosis percentages over this period will be computed and displayed.

### Confidence Interval on Next Day Close Price

Next, you will be prompted to enter the target confidence level (a value between 0 and 1). For example, for a 95% confidence level, enter 0.95.

The lower and upper bounds of the GARCH Model, EWMA Model, and Implied Volatility will be calculated and shown.

### Rolling Z-Score on Close Price

Finally, you will be prompted to enter the length of the rolling window for the Z-Score.

The script will plot the 50-day rolling Z-Score on the close price from the previous year.

The ADF test p-value and Hurst exponent will also be calculated and displayed.
