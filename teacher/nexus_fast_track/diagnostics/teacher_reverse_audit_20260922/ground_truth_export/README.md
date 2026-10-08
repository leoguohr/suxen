Teacher CAD50 lossless float-mesh targets

These files are exported from data/point50/training.pt in the supplied original results ZIP. The separately named teacher50_for_current_ae.zip was not supplied.
vertices are the exact FP32 normalized coordinates in that training cache. faces retain original cache order/winding. No quantization, normalization, welding, simplification, or vertex reorder was applied.
face_set and edge_index are derived undirected supervision; incidence_index has face nodes offset by the local vertex count. No octree or condition point cloud exists or is fabricated.
The teacher text features, weights, and predictions are not model inputs or supervision.
