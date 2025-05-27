# NCAAMB_tourney_simulation
NCAA_smalldata_tournamentSimulation: Choose your statistical model ensemble and run a bootstrapped tournament with your data.


-Use your choice of analyses to simulate the tournament, and see what performs best for the amount of data and variables you use. 
    - Use 'still_refining.py' with the cleaned and transformed data from the 2023-2024 season
    - The other scripts are for reference to data source and some VERY basic transformations to certain metrics
    - The model is not adjusted for conference but you can add a weight
-The ncaa website is a not the easiest to collect data from, so you will have to clean your input data thoroughly (beyond the web scraping script here).
-The test data is cleaned and transformed from the most recent year only (see above)
I threw this together in two days, and it is not efficient, nor comprehensive, but bootstrapped simulations performed well, and the emoji print messages are fun.

