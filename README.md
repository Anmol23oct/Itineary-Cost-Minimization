# ItinearyCostMinimization
For our project we implemented to A* algorithm and compared its performance against Dijkstra’s algorithm. The data we chose to evaluate was flight costs to and from various airports. The data was downloaded as a *.csv file from the U.S. Bureau of Transportation Statistics. We used the flight data from Quarter 1 of 2021, which gave us over three million flights across 429 airports. To simplify the data, we took the average price between each airport which reduced the graph space to 59,241 edges across 429 vertices. To make our A* heuristic more relevant, we then removed all flights with a cost under 5 dollars , leading to a final graph containing 429 vertices and 58,974 weighted, directional edges. The final graph represents the average cost of a flight between each airport during Q1 of 2021. The pruning and organization of the data was done in SQL and exported as *.csv file, which was then imported into Python.