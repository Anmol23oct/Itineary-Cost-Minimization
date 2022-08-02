# Description

In this Project, A* algorithm put into practice and evaluated its performance in comparison to Dijkstra's algorithm. We decided to analyze flight costs to and from various airports as our source of data. The information was obtained from the U.S. Bureau of Transportation Statistics as a.csv file. We used the first quarter of 2021's flight data, which provided us with information on almost three million flights across 429 airports. The graph space was reduced to 59,241 edges across 429 vertices by taking the average price between each airport in order to make the data more understandable. We next eliminated any flights with a price of less than $5 to make our A heuristic more applicable, leaving a final network with 429 vertices and 58,974 weighted, directional edges. The final graph displays the typical.


File Reference : dsa_project_final_01_04_14.pdf


# Code File
main.py
