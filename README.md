# SHAllenge searcher

This repository holds code to find a string of a specific format that gives a SHA256 hash with the lowest value. 
More specifically, it is to participate on [this online leaderboard](https://shallenge.quirino.net/).
It uses CUDA to run the hashing on the GPU.

The current version of the code uses some preprocessing of the SHA256 hashing function to speed up the search.
Preprocessing can be done, because the payload largely stays the same during the search.
The amount of preprocessing is optimized by putting the variable part of the payload at the end,
and always using a payload of 55 characters (the maximum amount characters that fit in one "chunk").
This allows us to precompute the first 13 rounds of the SHA256 function that processes a chunk.
On an RTX4090 it achieves a speed of 21.3 GH/s.

The number of rounds that is safe to precompute without changing the output was found using the script `constant_sha256_steps.py`.
It is a bit dirty and does more than it needs to, but it gets the job done.

The lowest value SHA256 hash I found is `00000000 000154ca 99d4bfc2 911a4f62 d046e935 5eaa7e1e bcb1575b 6be2c870`
for the payload `toteload/davidbos+dot+me/8OL6udTW`.

The lowest SHA1 hash found is `00000000 00005d2c 6785fd62 846f63a8 66569133` for the payload `toteload/davidbos+dot+me/sYGw1I08`.
You might wonder why I also have a SHA1 hash here when the leaderboard is for SHA256.
It is because I misread and originally wrote the code for SHA1 🙈

