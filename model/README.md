# Model card: Flight fare predictor

## Summary

Estimates the listed fare for a domestic flight in India. Built for exploration in the dashboard, not for booking decisions.

- **File:** `flight_fare_model.joblib` (scikit-learn pipeline)
- **Model:** Hist Gradient Boosting Regressor on log(1 + fare), tuned with a 3-fold randomized search
- **Inputs:** Airline, Class, Source, Destination, Departure time, Arrival time, Journey day, Stops, Duration (hours), Days left
- **Output:** predicted fare in ₹
- **Trained on:** 80% of 445,366 fare quotes, split stratified by class (356,292 train rows)

## Intended use

- Compare fares across airlines, classes, routes and booking windows.
- Get a rough estimate of what a trip is listed at.

Not for real-time pricing, buying decisions or any claim about what travellers actually paid.

## Data

- 7 cities, 9 airlines, 42 routes, departures 16 Jan to 6 Mar 2023.
- All prices were collected on a single day. This is narrow data, not small data: more rows from the same day add little, while fares across several days and peak periods would.
- Fare quotes, not bookings.
- Class mix is not representative of demand (about 28% Business, 16% Premium Economy).
- `Duration_in_hours` is total trip time including layovers. On the same routes, 1-stop trips take about 11 to 13 hours longer than non-stop, so duration reflects connection type, not flying time.

## Performance

The same model scored two ways:

| Metric | Random split | New flights (GroupKFold) |
|---|---|---|
| R² | 0.953 | 0.909 |
| MAE | ₹2,386 | ₹3,540 |
| MAPE | 10.8% | 16.8% |
| Within 10% of actual | 60.5% | 44.4% |
| Within 20% of actual | 84.2% | 69.5% |
| Same-flight lookup R² (baseline) | 0.922 | 0.799 |

- The same flight appears on many dates. In the random split, 99.9% of test rows belong to a flight that is also in training, so part of that score is recognition.
- In the new-flight test every flight is held out as a whole. Quote these numbers.
- A "later dates" split was also run, but it is not a true time split here (99.9% of its test flights also appear in training), so it is not reported.

### By class (new flights, 95% CI from resampling whole flights)

| Class | Rows | MAE (95% CI) | Average miss | Within 10% |
|---|---|---|---|---|
| Economy | 248,323 | ₹1,814 (1,776 to 1,850) | 18.1% | 41.8% |
| Premium Economy | 70,364 | ₹2,822 (2,728 to 2,918) | 17.0% | 45.1% |
| Business | 126,535 | ₹7,306 (7,098 to 7,534) | 14.2% | 49.4% |
| First | 144 | ₹20,866 (13,860 to 42,072) | 31.8% | 0.0% |

### By airline (new flights)

Compared by percentage error, because rupee errors scale with fare level and route mix.

- Most accurate: Air India (14.5% average miss, 53.1% within 10%).
- Least accurate: GO FIRST (21.5%, 35.0%).
- AllianceAir (12 flights) and StarAir (4 flights) have too few flights to judge.

### Systematic error

Average signed error on unseen flights, by class, airline, route and days-left bucket, stays within about +1% to +3% (slightly too high), with these exceptions:

- **First class: -31.8%.** The model underprices it by about a third.
- Chennai to Bangalore: +8.6% (priced too high).
- Slightly larger errors within 7 days of departure (+3.1%) than at 31+ days (+1.6%).

### Prediction intervals

Tested in the evaluation notebook with split conformal prediction on the log-fare scale (First pooled with Business). A target 80% range covered about 76% of real fares on unseen flights (Economy 76.3%, Business 76.7%, Premium Economy 73.3%). Treat a range as "about 3 in 4". Not shown in the app yet.

## Known weaknesses

- First class: 144 records from 4 flights. Estimates are rough and biased low.
- Single collection day: no holidays, peak periods or price changes over time.
- Trips combining an airline, route, class and stop count that never appear in the data are less reliable. The app warns about these.
- The model learns itinerary patterns through duration (which includes layovers), not flying time.

## Reproduce

Run `notebooks/02_evaluation.ipynb` next to `Cleaned_dataset.csv`. It rebuilds every table above, including the confidence intervals, bias checks and prediction intervals.