# Flight Fare Intelligence

Dashboard and fare predictor for domestic flights in India, built on 445,366 fare quotes.

**Live app:** https://flight-fare-intelligence.streamlit.app/

Data Science internship project at Naviotech Solution Pvt. Ltd.

---

## What it answers

| Question | Answer from the data |
|---|---|
| How much does cabin class matter? | Premium Economy is 1.6x, Business 5.1x and First 6.0x the Economy average (₹10,164) |
| Does booking late cost more? | Yes. Economy flights leaving within 3 days cost 49% more than those leaving 22+ days ahead (₹13,461 vs ₹9,054) |
| Which airlines are dearer? | In Economy, Vistara (₹12,082) and Air India (₹11,276) average 38 to 47% above Indigo (₹8,199). AkasaAir is cheapest at ₹3,570, partly because it flies short non-stop routes (average trip 1.6h vs 5.3h for Indigo). Airline averages reflect route mix |
| What is a fair price for my trip? | The Fare Predictor returns an estimate, the route median and the middle 50% of fares on that route. It is a guide: on flights it has not seen before, about 44% of estimates land within 10% of the listed fare |

## Dashboard

Sidebar filters: Class, Airline, From, To, Stops. Hero KPIs: Average fare, Total flights, Cheapest airline, Last-minute premium.

| Tab | Contents |
|---|---|
| Overview | Fare by class, fare vs days left (average or % change), fare distribution (histogram or box plot) |
| Airlines & Routes | Airline gap and spread, route heatmap, top 10 routes (costliest, cheapest, busiest), class mix |
| Timing, Stops & Duration | Departure and arrival time, weekday, stops, duration vs fare |
| Fare Predictor | Predicted fare vs route median and middle 50% of route fares, typical error for the class, fare vs days left, same trip on every airline |

### Feature highlights

- **Dynamic KPIs:** hero KPIs and tab KPI cards recalculate from the sidebar filters.
- **Dynamic insight boxes (🔎):** every chart has a takeaway generated from the filtered data, with changing numbers and names.
- **Smart Booking Tips:** four data-driven tips on the Overview, Airlines and Timing tabs, generated from the full dataset (nothing hardcoded). Airlines and Timing have a class selector, Economy by default.
- **Instant predictor:** all outputs update on any input change. Duration auto-fills with the typical value for the chosen airline, class, route and stops.
- **Honest error:** every prediction shows the typical miss for its class, measured on flights the model had not seen. First class carries a warning.
- **Smart warnings:** a "refine your comparison" note when several classes are selected, a low-confidence note when a trip combination is not in the data, and a no-results note when filters return no flights.
- **14 charts:** 12 analysis charts and 2 predictor charts, one chart per row, radio buttons only where charts share axes.
- **Mobile-friendly dark theme:** responsive KPI and tip grids, short city codes on charts.
- **Fast on 450k rows:** cached data and model, pre-binned histogram, sampled box and scatter plots.
- **Data and model note:** the sidebar "About this data" section explains the data scope, its limits and which model is in use. The sidebar also links to LinkedIn, GitHub and a fork page.

## Model

Hist Gradient Boosting on log(1 + fare), tuned with a 3-fold randomized search. 80/20 split stratified by class (356,292 train, 89,074 test).

The same model is scored two ways:

| Metric | Random split | New flights (GroupKFold) |
|---|---|---|
| R² | 0.953 | 0.909 |
| MAE | ₹2,386 | ₹3,540 |
| MAPE | 10.8% | 16.8% |
| Within 10% of actual | 60.5% | 44.4% |
| Within 20% of actual | 84.2% | 69.5% |
| Same-flight lookup R² (baseline) | 0.922 | 0.799 |

- **Random split:** the same flight appears on many dates, so nearly every test flight (99.9%) is also in training. Part of this score is the model recognising a flight it has seen.
- **New flights:** all rows of a flight stay on the same side of the split. This is the main number to quote. The model still beats the same-flight lookup (0.909 vs 0.799), so it learns real price patterns.
- Random split also gives RMSE ₹4,402 and a median error of 7.4%.

New-flight results by class, with 95% confidence intervals (flights resampled as a block):

| Class | Rows | MAE (95% CI) | Average miss | Within 10% |
|---|---|---|---|---|
| Economy | 248,323 | ₹1,814 (1,776 to 1,850) | 18.1% | 41.8% |
| Premium Economy | 70,364 | ₹2,822 (2,728 to 2,918) | 17.0% | 45.1% |
| Business | 126,535 | ₹7,306 (7,098 to 7,534) | 14.2% | 49.4% |
| First | 144 | ₹20,866 (13,860 to 42,072) | 31.8% | 0.0% |

- **First class is not reliable:** 144 records from 4 flights, and the model underprices it by about a third.
- **Airlines:** Air India is the most accurate (14.5% average miss), GO FIRST the least (21.5%). Airlines are compared by percentage error because of route mix.
- **Bias check:** by class, airline, route and days-left bucket, average errors stay within a few percent, except First class (-31.8%).
- **Prediction intervals:** tested in the notebook with conformal prediction. An 80% range covered about 76% of real fares on unseen flights. Not shown in the app yet.

Compared against baseline, Linear, Ridge, Decision Tree and Random Forest. Class is the strongest feature, then duration, route and airline.

Modelling: [`notebooks/01_flight_fare_prediction.ipynb`](notebooks/01_flight_fare_prediction.ipynb). Full evaluation: [`notebooks/02_evaluation.ipynb`](notebooks/02_evaluation.ipynb). Model details: [`model/README.md`](model/README.md).

## Data

`Cleaned_dataset.csv`: 7 cities, 9 airlines, 42 routes, departures 16 Jan to 6 Mar 2023.

Cleaning: strip spaces, convert numerics, drop NaN and duplicates (6,722 rows), drop `Date_of_journey` and `Flight_code`, add `Route`.

Prices are fare quotes collected on a single day, not bookings. Data and model limitations are listed in [`model/README.md`](model/README.md).

## Run locally

```bash
git clone https://github.com/Pratham719/flight-fare-intelligence.git
cd flight-fare-intelligence
pip install -r requirements.txt
streamlit run app.py
```

Pinned versions: joblib 1.5.3, numpy 2.4.1, pandas 2.3.3, plotly 6.7.0, scikit-learn 1.8.0, scipy 1.17.1, streamlit 1.55.0.

## Project structure

```
flight-fare-intelligence/
├── app.py                          Streamlit dashboard and predictor
├── Cleaned_dataset.csv             Dataset
├── requirements.txt                Pinned dependencies
├── LICENSE
├── README.md
├── .streamlit/
│   └── config.toml                 App settings
├── notebooks/
│   ├── 01_flight_fare_prediction.ipynb   EDA and modelling
│   └── 02_evaluation.ipynb               New-flight split, confidence intervals, bias, prediction intervals
├── Documentation/
│   ├── presentation.pptx
│   ├── presentation.pdf
│   ├── report.docx
│   └── report.pdf
└── model/
    ├── flight_fare_model.joblib    Trained model
    └── README.md                   Model card
```

## Documentation

- [Presentation (PDF)](Documentation/presentation.pdf)
- [Project report (PDF)](Documentation/report.pdf)

## Author

Pratham Rangoonwala | Data Science Intern | Naviotech Solution Pvt. Ltd.

LinkedIn: [@pratham-ds](https://www.linkedin.com/in/pratham-ds)