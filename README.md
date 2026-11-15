# kmeans-at-scale-benchmarks

Independent, reproducible benchmarks of k-means training when k is large: Lloyd, k-means++, mini-batch, e-means, o-means and whatever comes next. Every comparison lives in its own dated folder with its code, raw results and figures, and links to the write-up on saliltrehan.com.

Results here are one person's measurements on rented GPUs and a Mac mini. They are not replications of any vendor's numbers unless a folder says so.

## Comparisons

| Folder | Question | Write-up |
|---|---|---|
| o-means-2026-11 | Released o-means vs e-means at equal epochs on SIFT and a hard synthetic mixture, A10G, L40S, A100 | o-means vs k-means: 2.4x Faster at 16,384 Centroids (saliltrehan.com/post/o-means-vs-kmeans) |

## Shared code

- common/guard.py : single-job resource guard for a shared machine (wall-clock cap, RSS cap, free-memory floor, heartbeat stall). Every long job in this repo runs under a hard time cap and a monitor.

## Adding a comparison

Add a folder named algorithm-or-topic-YYYY-MM with its own README (what was tested, what was not, versions, how to reproduce), raw results and figures, then add one row to the table above. Do not edit results in older folders.

## Licence

MIT.
