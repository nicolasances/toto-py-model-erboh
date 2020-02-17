# Model erboh
The `erboh` model classifies expenses as monthly expenses based on the historical data (looking at the time series and finding expenses that have some repetitive pattern)

A description of this model's behaviour can be found at this [Toto Wiki page](https://github.com/nicolasances/guides/wiki/Model-Design:-erboh).

## Environment
This model runs as a containerized microservice. 

It needs the following environment variables: 
 * **TOTO_API_AUTH**: API authentication param to be used when authenticating to Toto APIs
 * **TOTO_HOST**: the host where APIs are reachable 
 * **TOTO_TMP_FOLDER**: a folder to store tmp files - This should be set in the Dockerfile
 * **TOTO_EVENTS_GCP_PROJECT_ID**: the Google Project Id for events in the current environment

## Predictions
This model generates predictions in two ways: 
 * **Batch**: will generate predictions for all expenses that do not have a `monthly` field set
 * **single**: will generate a prediction on demand for a single expense (indenpendently of whether that expense has the `monthly` field set)

In both approaches, the model generates files and stores them in a temporary folder under:
```
> {TOTO_TMP_FOLDER}/erboh/<uuid>/
```
Under that folder, the model will save:
 * `history` - a file with the relevant downloaded history
 * `features` - a file with the built features for the model
 * `predictions` (only in the batch inference) - a file with the generated predictions