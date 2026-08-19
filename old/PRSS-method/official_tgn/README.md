# Exact upstream TGN source

`source/` is a `git archive` of the bundled `twitter-research/tgn` repository at commit:

`d55bbe678acabb9fc3879c408fd1f2e15919667c`

`train_supervised.py` at this directory is copied byte-for-byte from `source/train_supervised.py`.
The PRSS implementation does not edit files inside `source/`; it wraps the recursive embedding module from outside.
