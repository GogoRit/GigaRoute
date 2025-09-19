### **Phase 1: Data Preprocessing and Graph Construction**

**Objective:** Transform the raw NYC OpenStreetMap data into a highly efficient, GPU-friendly graph format.

**Tools:**

  * **Language:** C++
  * **Libraries:** Libosmium, CMake
  * **Input Data:** `new-york-latest.osm.pbf`

**Approach & Key Features**
The project uses a two-pass processing pipeline to handle the large-scale dataset efficiently.

  * **Pass 1: ID Mapping:** Iterated through all road segments (`ways` with a "highway" tag) and mapped 11,726,672 unique 64-bit OSM node IDs to a contiguous 32-bit ID space. This is crucial for memory efficiency.
  * **Pass 2: Graph Construction:** Used the ID map to build the final graph representation. Edge weights were calculated as the **haversine distance** (in meters) and all roads were treated as **bidirectional**.

**Output: Compressed Sparse Row (CSR) Graph**
The final output is a sparse graph stored in the CSR format, optimized for GPU parallel processing. This data is saved to a binary file to ensure fast loading in subsequent phases.

  * **File:** `nyc_graph.bin`
  * **Format:** The file contains three contiguous arrays: `row_pointers`, `column_indices`, and `values`.

**Final Statistics**
The program successfully processed the entire NYC road network and generated the following statistics.

  * **Nodes:** 11,726,672 unique intersections
  * **Edges:** 25,290,318 bidirectional road segments
  * **Final File Size:** 350,390,516 bytes (\~350 MB)

**Console Log**

```
Pass 1: Creating node ID map...
Done. Found 11726672 unique road nodes.
Pass 2: Building graph data structures...
Building CSR representation...
Done. Graph has 25290318 edges.
Writing graph to nyc_graph.bin...
Graph successfully saved to nyc_graph.bin
Summary:
  Nodes: 11726672
  Edges: 25290318
  File size: 350390516 bytes
```

-----

### **Phase 2: CPU Baseline and Benchmarking**

**Objective:** Implement a single-threaded CPU algorithm to establish a performance baseline for future GPU acceleration.

**Tools:**

  * **Language:** C++
  * **Standard Library:** `std::vector`, `std::priority_queue`, `std::chrono`
  * **Input Data:** `nyc_graph.bin`

**Approach & Key Features**
The project successfully implemented a two-part CPU-based solution:

1.  **Graph Parser:** A `GraphParser` class was created to efficiently load the `nyc_graph.bin` file's CSR arrays into memory.
2.  **Dijkstra's Algorithm:** A `DijkstraPathfinder` class was implemented using a `std::priority_queue` to find the shortest path between two nodes. It includes a timer for performance measurement.

**Final Statistics**
The program was run on the full NYC graph to obtain baseline performance metrics.

**Graph Statistics:**

```
=== Graph Statistics ===
Nodes: 11726672
Edges: 25290318
Average degree: 2.15665
Min degree: 1
Max degree: 10
Min edge weight: 0 meters
Max edge weight: 3105.58 meters
```

**Console Log & Performance Metrics:**

**Test 1: Long Path (Random Nodes)**

```
=== Path Finding Result ===
Start node: 2931668
End node: 5863336
Path found: Yes
Total distance: 96095.3 meters
Distance in kilometers: 96.0953 km
Path length (nodes): 1474
Nodes visited during search: 2493400
Computation time: 1461.64 ms
Path (first 10 and last 10 nodes): 2931668 -> 2931667 -> 2931666 -> 2931665 -> 2860940 -> 2860939 -> 2860938 -> 2860937 -> 2860936 -> 2860935 ...  -> 5863345 -> 5863344 -> 5863343 -> 5863342 -> 5863341 -> 5863340 -> 5863339 -> 5863338 -> 5863337 -> 5863336
```

**Test 2: Short Path (Node 0 to Node 100)**

```
=== Path Finding Result ===
Start node: 0
End node: 100
Path found: Yes
Total distance: 380360 meters
Distance in kilometers: 380.36 km
Path length (nodes): 4953
Nodes visited during search: 4232726
Computation time: 2394.97 ms
Path (first 10 and last 10 nodes): 0 -> 4412508 -> 74 -> 73 -> 72 -> 71 -> 5311431 -> 5311430 -> 5311429 -> 5311428 ...  -> 4730767 -> 4730768 -> 4730769 -> 4730770 -> 95 -> 96 -> 97 -> 98 -> 99 -> 100
```