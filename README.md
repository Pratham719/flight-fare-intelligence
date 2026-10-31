# Flight Fare Intelligence

Dashboard and fare predictor for domestic flights in India, built on 445,366 fares.

**Live app:** https://flight-fare-intelligence.streamlit.app/

Data Science internship project at Naviotech Solution Pvt. Ltd.

---

## What it answers

| Question | Answer from the data |
|---|---|
| How much does cabin class matter? | Premium Economy is 1.6x, Business 5.1x and First 6.0x the Economy average (₹10,164) |
| Does booking late cost more? | Yes. Economy flights leaving within 3 days cost 49% more than those leaving 22+ days ahead (₹13,461 vs ₹9,054) |
| Which airlines are dearer? | In Economy, Vistara (₹12,082) and Air India (₹11,276) average 38 to 47% above Indigo (₹8,199). AkasaAir is cheapest at ₹3,570 |
| What is a fair price for my trip? | The Fare Predictor returns an estimate, the route median and the typical range |

## Dashboard

Sidebar filters: Class, Airline, From, To, Stops. Hero KPIs: Average fare, Total flights, Cheapest airline, Last-minute premium.

| Tab | Contents |
|---|---|
| Overview | Fare by class, fare vs days left (average or % change), fare distribution (histogram or box plot) |
| Airlines & Routes | Airline gap and spread, route heatmap, top 10 routes (costliest, cheapest, busiest), class mix |
| Timing, Stops & Duration | Departure and arrival time, weekday, stops, duration vs fare |
| Fare Predictor | Predicted fare vs route median and typical range, fare vs days left, same trip on every airline |

### Feature highlights

- **Dynamic KPIs:** hero KPIs and tab KPI cards recalculate from the sidebar filters.
- **Dynamic insight boxes (🔎):** every chart has a takeaway generated from the filtered data, with changing numbers and names.
- **Smart Booking Tips:** four data-driven tips on the Overview, Airlines and Timing tabs, generated from the full dataset (nothing hardcoded). Airlines and Timing have a class selector, Economy by default.
- **Instant predictor:** all outputs update on any input change. Duration auto-fills with the typical value for the chosen airline, class, route and stops.
- **Smart warnings:** a "refine your comparison" note when several classes are selected, a low-confidence note when a trip combination is not in the data, and a no-results note when filters return no flights.
- **14 charts:** 12 analysis charts and 2 predictor charts, one chart per row, radio buttons only where charts share axes.
- **Mobile-friendly dark theme:** responsive KPI and tip grids, short city codes on charts.
- **Fast on 450k rows:** cached data and model, pre-binned histogram, sampled box and scatter plots.
- **Model transparency:** the sidebar shows whether the saved model or the built-in fallback is in use.

## Model

Hist Gradient Boosting on log(1 + fare), tuned with a 3-fold randomized search. 80/20 split stratified by class (356,292 train, 89,074 test).

| Metric (test set) | Value |
|---|---|
| R² | 0.953 |
| MAE | ₹2,386 |
| RMSE | ₹4,402 |
| Median error | 7.4% |
| Within 11% of actual | 64.0% |
| Within 20% of actual | 84.2% |

Compared against baseline, Linear, Ridge, Decision Tree and Random Forest. Class is the strongest feature, then duration, route and airline.

## Data

`Cleaned_dataset.csv`: 7 cities, 9 airlines, 42 routes, departures 16 Jan to 6 Mar 2023.

Cleaning: strip spaces, convert numerics, drop NaN and duplicates (6,722 rows), drop `Date_of_journey` and `Flight_code`, add `Route`.

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
├── flight_fare_prediction.ipynb    EDA, modelling, evaluation
├── requirements.txt                Pinned dependencies
├── LICENSE
├── README.md
├── .streamlit/
│   └── config.toml                 Dark theme settings
├── Documentation/
│   ├── flight_fare_presentation.pptx
│   ├── flight_fare_presentation.pdf
│   ├── flight_fare_report.docx
│   └── flight_fare_report.pdf
└── model/
    └── flight_fare_model.joblib    Trained model
```

## Documentation

- [Presentation (PDF)](Documentation/presentation.pdf)
- [Project report (PDF)](Documentation/report.pdf)

## Author

Pratham Rangoonwala | Data Science Intern | Naviotech Solution Pvt. Ltd.

GitHub: [@Pratham719](https://github.com/Pratham719)